package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"math"
	"math/bits"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"sort"
	"syscall"
	"time"
	"unsafe"
)

/*
#include <stdio.h>
#include <arm_neon.h>

void dump(char *msg, uint8x16_t r) {
	uint8_t buf[16];
	vst1q_u8(buf, r);
	fprintf(stdout, "%s", msg);
	for (int i = 0; i < 16; i++) {
		fprintf(stdout, " %02x", buf[i]);
	}
	fprintf(stdout, "\n");
}

void split(uint8_t *mem, int length, uint8_t *out) {
	// Each 8-byte chunk of memory can have at most 1 newline. Define a mask to use
	// separately on the high and low bits which identifies the byte+1 that contains
	// the newline.
	const uint64_t maskData = (1ULL << 0) | (2ULL << 8) | (3ULL << 16) | (4ULL << 24) |
		(5ULL << 32) | (6ULL << 40) | (7ULL << 48) | (8ULL << 56);
	uint8x16_t mask = vreinterpretq_u8_u64(vdupq_n_u64(maskData));
	uint8x16_t newline = vdupq_n_u8((uint8_t)'\n');
	int count = 0;
	for (uint8_t *end = mem + length; mem < end; mem += 16, out += 2) {
		uint8x16_t value = vld1q_u8(mem);
		// dump("value:   ", value);
		uint8x16_t compared = vceqq_u8(value, newline);
		// dump("compared:", compared);
		uint8x16_t masked = vandq_u8(compared, mask);
		// dump("masked:  ", masked);
		out[0] = vaddv_u8(vget_low_u8(masked));
		out[1] = vaddv_u8(vget_high_u8(masked));
		// fprintf(stdout, "out: %02x %02x\n", out[0], out[1]);
	}
}

*/
import "C"

type measurement struct {
	min   int32
	max   int32
	sum   int32
	count int32
}

func main() {
	if robinHoodEntrySize != unsafe.Sizeof((robinHoodEntry{})) {
		panic("not reached")
	}

	name := "Run"
	if os.Getenv("BENCH") == "1" {
		defer func(start time.Time) {
			fmt.Printf("Benchmark%s\t1\t%d ns/op\n", name, time.Since(start))
		}(time.Now())
	}

	if len(os.Args) > 3 {
		log.Fatalf("Usage: %s [<measurements>] [profile]", os.Args[0])
	}

	path := "measurements.txt"
	if len(os.Args) >= 2 {
		path = os.Args[1]
	}

	if len(os.Args) == 3 {
		prof, err := os.Create(os.Args[2])
		if err != nil {
			panic(err)
		}
		if err = pprof.StartCPUProfile(prof); err != nil {
			panic(err)
		}
		defer func() {
			pprof.StopCPUProfile()
		}()
	}

	parallelism := runtime.GOMAXPROCS(0) + 2
	runtime.GOMAXPROCS(parallelism)
	debug.SetGCPercent(-1)

	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		panic(err)
	}
	size := fi.Size()
	if size <= 0 {
		panic("invalid file size")
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		panic(err)
	}

	switch os.Getenv("MODE") {
	case "count-lines-stdlib":
		name = "CountLines"
		chunks := split(data, parallelism)
		results := make(chan int, len(chunks))
		for _, chunk := range chunks {
			go func(chunk []byte) {
				results <- bytes.Count(chunk, []byte("\n"))
			}(chunk)
		}
		sum := 0
		for i := 0; i < len(chunks); i++ {
			sum += <-results
		}
		fmt.Printf("%d\n", sum)

	case "count-lines":
		name = "CountLines"
		chunks := split(data, parallelism)
		results := make(chan int, len(chunks))
		for _, chunk := range chunks {
			go func(chunk []byte) {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				results <- countLines(chunk)
			}(chunk)
		}
		sum := 0
		for i := 0; i < len(chunks); i++ {
			sum += <-results
		}
		fmt.Printf("%d\n", sum)

	case "count-lines-c":
		name = "CountLinesC"
		chunks := split(data, parallelism)
		results := make(chan int, len(chunks))
		for _, chunk := range chunks {
			go func(chunk []byte) {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				results <- countLinesC(chunk)
			}(chunk)
		}
		sum := 0
		for i := 0; i < len(chunks); i++ {
			sum += <-results
		}
		fmt.Printf("%d\n", sum)

	default:
		name = "Process"
		chunks := split(data, parallelism)
		results := make(chan *robinHoodMap, len(chunks))
		for _, chunk := range chunks {
			go func(chunk []byte) {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				m := newRobinHoodMap(8192)
				process(chunk, m)
				results <- m
			}(chunk)
		}

		cities := <-results
		for i := 1; i < len(chunks); i++ {
			r := <-results
			r.Iterate(func(hash uint64, name robinHoodKey, rm *measurement) {
				cities.Upsert(hash, name, func(m *measurement) {
					if m.count == 0 {
						*m = *rm
					} else {
						m.min = min(m.min, rm.min)
						m.max = max(m.max, rm.max)
						m.sum += rm.sum
						m.count += rm.count
					}
				})
			})
		}

		type nameHash struct {
			hash uint64
			name robinHoodKey
		}
		var names []nameHash
		cities.Iterate(func(hash uint64, name robinHoodKey, _ *measurement) {
			// names = append(names, name)
			names = append(names, nameHash{
				hash: hash,
				name: name,
			})
		})
		sort.Slice(names, func(i, j int) bool {
			a := unsafe.String((*byte)(unsafe.Pointer(&names[i].name)), unsafe.Sizeof((robinHoodKey{})))
			b := unsafe.String((*byte)(unsafe.Pointer(&names[j].name)), unsafe.Sizeof((robinHoodKey{})))
			return a < b
		})

		for _, n := range names {
			cities.Upsert(n.hash, n.name, func(m *measurement) {
				s := unsafe.Slice((*byte)(unsafe.Pointer(&n.name.data[0])), unsafe.Sizeof((robinHoodKey{})))
				for i := range s {
					if s[i] == 0 {
						s = s[:i]
						break
					}
				}
				fmt.Printf("%s=%.1f/%.1f/%.1f\n", s,
					float64(m.min)/10, (float64(m.sum)/float64(m.count))/10, float64(m.max)/10)
			})
		}
	}
}

func split(data []byte, n int) [][]byte {
	chunks := make([][]byte, n)
	chunkSize := len(data) / n
	var start int
	for i := range chunks {
		end := (i + 1) * chunkSize
		if end >= len(data) {
			end = len(data)
		} else if j := bytes.IndexByte(data[end:], '\n'); j == -1 {
			end = len(data)
		} else {
			end += j + 1
		}
		chunks[i] = data[start:end]
		start = end
	}
	return chunks
}

var tempTable [256]int16
var tempLengthTable [16]uint8

func init() {
	// There are 4 cases to parse:
	//   ;-xx.x
	//    ;xx.x
	//    ;-x.x
	//    y;x.x
	//
	// The "x.x" can be parsed simply by extracting the two bytes.
	//
	// The first 2 bytes fall into 4 categories: "-x", ";x", ";-" and "y;".
	// The digits, semicolon, and minus sign can all be distinguished by
	// looking at the low 4 bits of each byte. Combining the low 4 bits from
	// the first 2 bytes results in an 8-bit value that is indexed into a 256
	// element table.

	for v := int32(0); v <= 9; v += 1 {
		tempTable[v|(('-'&0xf)<<4)] = int16(-v*100 - 1)
		tempTable[v|((';'&0xf)<<4)] = int16(v * 100)
	}
	tempTable[((';'&0xf)<<4)|('-'&0xf)] = int16(-1)
	for v := 0; v <= 0xf; v++ {
		tempTable[(v<<4)|(';'&0xf)] = 0
	}
	for i := range tempLengthTable {
		tempLengthTable[i] = 4
	}
	tempLengthTable['-'&0xf] = 6
	tempLengthTable[';'&0xf] = 5
}

func countLines(data []byte) int {
	buf := make([]byte, 4096)
	count := 0
	sum := 0
	for j, length := 0, len(data); j < length; {
		n := copy(buf[:cap(buf)], data[j:])
		buf = buf[:n]
		last := bytes.LastIndexByte(buf, '\n')
		buf = buf[:last+1]
		j += last + 1

		ptr := unsafe.Pointer(unsafe.SliceData(buf))
		start := 0
		for i := 0; i < len(buf); {
			v := *((*uint64)(unsafe.Pointer(uintptr(ptr) + uintptr(i))))
			// https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
			q := v ^ 0x0a0a0a0a0a0a0a0a
			r := (((q - 0x0101010101010101) &^ q) & 0x8080808080808080)
			if r == 0 {
				i += 8
				continue
			}

			t := bits.TrailingZeros64(r) / 8
			end := i + t
			start = end + 1
			_ = start
			i += 8
			count++

			if false {
				d1 := int32((*(*byte)(unsafe.Pointer(uintptr(ptr) + uintptr(end-1))) & 0xf)) +
					10*(int32(*(*byte)(unsafe.Pointer(uintptr(ptr) + uintptr(end-3)))&0xf))
				d2 := (*(*uint8)(unsafe.Pointer(uintptr(ptr) + uintptr(end-4))) & 0xf) |
					((*(*uint8)(unsafe.Pointer(uintptr(ptr) + uintptr(end-5))) & 0xf) << 4)
				v := int32(tempTable[d2])
				var temp int32
				if v < 0 {
					temp = v + 1 - d1
				} else {
					temp = v + d1
				}
				sum += int(temp)
			}

			if false {
				d0 := *(*byte)(unsafe.Pointer(uintptr(ptr) + uintptr(end-1)))
				d2 := *(*byte)(unsafe.Pointer(uintptr(ptr) + uintptr(end-3)))
				d3 := *(*byte)(unsafe.Pointer(uintptr(ptr) + uintptr(end-4)))
				temp := 10*int(d2&0xf) + int(d0&0xf)
				if d3 == '-' {
					temp = -temp
				} else if d3 != ';' {
					temp += 100 * int(d3&0xf)
					if *(*byte)(unsafe.Pointer(uintptr(ptr) + uintptr(end-5))) == '-' {
						temp = -temp
					}
				}
				sum += temp
			}
		}
	}
	fmt.Fprint(io.Discard, sum)
	return count
}

// Using Neon instructions is ~30% faster than the pure-Go version for finding
// newlines.
func countLinesC(data []byte) int {
	zeros := make([]byte, 16)
	buf := make([]byte, 16384)
	bufPtr := unsafe.Pointer(unsafe.SliceData(buf))
	out := make([]uint8, len(buf)/8)
	outPtr := unsafe.Pointer(unsafe.SliceData(out))
	count := 0
	sum := 0

	for j, length := 0, len(data); j < length; {
		// We need to copy into buf in order to ensure the data is 16-byte
		// aligned which is needed for the neon instructions. The overhead of
		// this copy is minimal and more than outweighed by using the neon
		// instructions.
		n := copy(buf[:cap(buf)], data[j:])
		last := bytes.LastIndexByte(buf[:n], '\n')
		j += last + 1

		m := (last + 1 + 15) &^ 15
		buf = buf[:m]
		copy(buf[last+1:], zeros)

		outLen := len(buf) / 8
		C.split((*C.uint8_t)(unsafe.Pointer(&buf[0])), (C.int)(len(buf)), (*C.uint8_t)(unsafe.Pointer(&out[0])))

		for i := 0; i < outLen; i++ {
			bit := int(*((*uint8)(unsafe.Pointer(uintptr(outPtr) + uintptr(i)))))
			if bit == 0 {
				continue
			}
			count++

			if false {
				index := i*8 + bit - 1
				d5 := (*(*uint8)(unsafe.Pointer(uintptr(bufPtr) + uintptr(index-5))) & 0xf)
				sum += index - int(tempLengthTable[d5])

				d1 := int((*(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(index-1))) & 0xf)) +
					10*(int(*(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(index-3)))&0xf))
				d2 := (*(*uint8)(unsafe.Pointer(uintptr(bufPtr) + uintptr(index-4))) & 0xf) | (d5 << 4)
				temp := int(tempTable[d2])
				if temp < 0 {
					temp += 1 - d1
				} else {
					temp += d1
				}
				sum += temp
			}

			if false {
				index := i*8 + bit - 1
				d0 := *(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(index-1)))
				d2 := *(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(index-3)))
				d3 := *(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(index-4)))
				temp := 10*int(d2&0xf) + int(d0&0xf)
				if d3 == '-' {
					temp = -temp
				} else if d3 != ';' {
					temp += 100 * int(d3&0xf)
					if *(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(index-5))) == '-' {
						temp = -temp
					}
				}
				sum += temp
			}
		}
	}
	fmt.Fprint(io.Discard, sum)
	return count
}

func process(data []byte, m *robinHoodMap) {
	// FNV hash base and hash mul
	const hashBase uint64 = 14695981039346656037
	const hashMul uint64 = 1099511628211

	mapEntries := unsafe.Pointer(unsafe.SliceData(m.entries))
	buf := make([]byte, 16384)
	bufPtr := unsafe.Pointer(unsafe.SliceData(buf))
	out := make([]uint8, len(buf)/8)
	outPtr := unsafe.Pointer(unsafe.SliceData(out))
	zeros := make([]byte, 16)

	for j, length := 0, len(data); j < length; {
		n := copy(buf[:cap(buf)], data[j:])
		last := bytes.LastIndexByte(buf[:n], '\n')
		j += last + 1

		n = (last + 1 + 15) &^ 15
		buf = buf[:n]
		copy(buf[last+1:], zeros)

		outLen := len(buf) / 8
		C.split((*C.uint8_t)(bufPtr), (C.int)(len(buf)), (*C.uint8_t)(outPtr))

		start := 0

	outer:
		for i := 0; i < outLen; i++ {
			bit := int(*((*uint8)(unsafe.Pointer(uintptr(outPtr) + uintptr(i)))))
			if bit == 0 {
				continue
			}

			lineEnd := i*8 + bit - 1

			var temp int
			var end int

			if false {
				// NB: Table-based temperature parser. See init() for the
				// table construction. This is slightly slower than the
				// temperature parsing approach below on go1.21 and slightly
				// (1-2%) faster on go1.22.
				d5 := (*(*uint8)(unsafe.Pointer(uintptr(bufPtr) + uintptr(lineEnd-5))) & 0xf)
				end = lineEnd - int(tempLengthTable[d5])

				d1 := int((*(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(lineEnd-1))) & 0xf)) +
					10*(int(*(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(lineEnd-3)))&0xf))
				d2 := (*(*uint8)(unsafe.Pointer(uintptr(bufPtr) + uintptr(lineEnd-4))) & 0xf) | (d5 << 4)
				temp = int(tempTable[d2])
				if temp < 0 {
					temp += 1 - d1
				} else {
					temp += d1
				}
			}

			if true {
				// Assume the data is well-formed, there are 4 possibilies for
				// temperatures in the range -99.9 to 99.9: -xx.x, -x.x, x.x, xx.x
				d0 := *(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(lineEnd-1)))
				d2 := *(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(lineEnd-3)))
				d3 := *(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(lineEnd-4)))
				temp = 10*int(d2&0xf) + int(d0&0xf)
				if d3 == '-' {
					temp = -temp
					end = lineEnd - 5
				} else if d3 != ';' {
					temp += 100 * int(d3&0xf)
					if *(*byte)(unsafe.Pointer(uintptr(bufPtr) + uintptr(lineEnd-5))) == '-' {
						temp = -temp
						end = lineEnd - 6
					} else {
						end = lineEnd - 5
					}
				} else {
					end = lineEnd - 4
				}
			}

			h := hashBase
			j := start
			cityStr := unsafe.String((*byte)(unsafe.Pointer(uintptr(bufPtr)+uintptr(start))), end-start)
			for j = start; j+8 <= end; j = j + 8 {
				v := *((*uint64)(unsafe.Pointer(uintptr(bufPtr) + uintptr(j))))
				h = (h ^ (v & 0xffff)) * hashMul
				h = (h ^ ((v >> 16) & 0xffff)) * hashMul
				h = (h ^ ((v >> 32) & 0xffff)) * hashMul
				h = (h ^ (v >> 48)) * hashMul
			}
			if j < end {
				v := *((*uint64)(unsafe.Pointer(uintptr(bufPtr) + uintptr(j))))
				v &= ((uint64(1) << (8 * uint((end-start)%8))) - 1)
				h = (h ^ (v & 0xffff)) * hashMul
				h = (h ^ ((v >> 16) & 0xffff)) * hashMul
				h = (h ^ ((v >> 32) & 0xffff)) * hashMul
				h = (h ^ (v >> 48)) * hashMul
			}
			start = lineEnd + 1

			// fmt.Printf("foo: %016x %x\n", h, city)

			// Fast-path: expect the entry to already exist in the hash table.
			i := int(h >> m.shift)
			for end := i + int(m.maxDist); i < end; i++ {
				e := ((*robinHoodEntry)(unsafe.Pointer(uintptr(mapEntries) + robinHoodEntrySize*uintptr(i))))
				if h == e.hash /* cityStr == e.key.AsString() */ {
					// Entry already exists: update.
					m := &e.value
					m.min = min(m.min, int32(temp))
					m.max = max(m.max, int32(temp))
					m.sum += int32(temp)
					m.count++
					continue outer
				}
			}

			var city robinHoodKey
			city.len = uint32(len(cityStr))
			copy(unsafe.Slice(&city.data[0], len(city.data)), cityStr)

			// Slow-path: insert a new entry in the hash table.
			m.Upsert(h, city, func(m *measurement) {
				m.min = min(m.min, int32(temp))
				m.max = max(m.max, int32(temp))
				m.sum += int32(temp)
				m.count++
			})

			// The map's keys and entries slices can only change on insertion.
			mapEntries = unsafe.Pointer(unsafe.SliceData(m.entries))
		}
	}

	// maxDist := uint32(0)
	// for i := range m.entries {
	// 	e := &m.entries[i]
	// 	maxDist = max(maxDist, e.dist)
	// }
	// fmt.Printf("max-dist=%d\n", maxDist)
}

const robinHoodEntrySize = 128

type robinHoodKey struct {
	data [100]byte
	len  uint32
}

func (k *robinHoodKey) AsString() string {
	return unsafe.String(&k.data[0], int(k.len))
}

type robinHoodEntry struct {
	hash  uint64
	key   robinHoodKey
	value measurement
}

type robinHoodMap struct {
	entries []robinHoodEntry
	dist    []uint32
	size    uint32
	shift   uint32
	maxDist uint32
}

func newRobinHoodMap(initialCapacity int) *robinHoodMap {
	m := &robinHoodMap{}
	if initialCapacity < 1 {
		initialCapacity = 1
	}
	targetSize := 1 << (uint(bits.Len(uint(2*initialCapacity-1))) - 1)
	m.rehash(uint32(targetSize))
	return m
}

func (m *robinHoodMap) rehash(size uint32) {
	oldEntries := m.entries

	m.size = size
	m.shift = uint32(64 - bits.Len32(m.size-1))
	m.maxDist = max(uint32(bits.Len32(size)), 4)
	m.entries = make([]robinHoodEntry, size+m.maxDist)
	m.dist = make([]uint32, size+m.maxDist)
	for i := range m.entries {
		e := &m.entries[i]
		e.value.min = math.MaxInt32
		e.value.max = math.MinInt32
	}

	for i := range oldEntries {
		if e := &oldEntries[i]; e.value.count > 0 {
			m.Upsert(e.hash, e.key, func(m *measurement) {
				*m = e.value
			})
		}
	}
}

func (m *robinHoodMap) Upsert(hash uint64, key robinHoodKey, f func(v *measurement)) {
	maybeExists := true
	var dist uint32
	n := robinHoodEntry{hash: hash, key: key}
	for i := hash >> m.shift; ; i++ {
		e := &m.entries[i]
		if maybeExists && key == e.key {
			// Entry already exists: overwrite.
			f(&m.entries[i].value)
			return
		}

		if e.value.count == 0 {
			// Found an empty entry: insert here.
			*e = n
			m.dist[i] = dist
			if maybeExists {
				f(&e.value)
			}
			return
		}

		if m.dist[i] < dist {
			// Swap the new entry with the current entry because the current is
			// rich. We then continue to loop, looking for a new location for the
			// current entry. Note that this is also the not-found condition for
			// retrieval, which means that "k" is not present in the map.
			n, *e = *e, n
			dist, m.dist[i] = m.dist[i], dist
			if maybeExists {
				f(&e.value)
				maybeExists = false
			}
		}

		// The new entry gradually moves away from its ideal position.
		dist++

		// If we've reached the max distance threshold, grow the table and restart
		// the insertion.
		if dist == m.maxDist {
			m.rehash(2 * m.size)
			dist = 0
			i = (hash >> m.shift) - 1
		}
	}
}

func (m *robinHoodMap) Iterate(f func(h uint64, k robinHoodKey, m *measurement)) {
	for i := range m.entries {
		if e := &m.entries[i]; e.value.count > 0 {
			f(e.hash, e.key, &e.value)
		}
	}
}
