package main

import (
	"bytes"
	"fmt"
	"math/bits"
	"os"
	"runtime"
	"sort"
	"sync"
	"unsafe"
)

const chunkSize = 1024 * 1024

var ch chan fileRange
var file *os.File
var wg sync.WaitGroup

type fileRange struct {
	ofs, len int64
}

type partial struct {
	ofs   int64
	start bool
	str   string
}

type worker struct {
	h        hashTable
	partials []partial
}

func (w *worker) run() {
	buf := make([]byte, chunkSize)
	partials := make([]partial, 0, 100)

	for r := range ch {
		b := buf[0:r.len]
		_, err := file.ReadAt(b, r.ofs)
		checkErr(err)
		firstLineEnd := bytes.IndexByte(b, '\n')
		partials = append(partials, partial{
			ofs:   r.ofs,
			start: false,
			str:   string(b[:firstLineEnd+1]),
		})
		lastLineEnd := bytes.LastIndexByte(b, '\n')
		if lastLineEnd < len(b)-1 {
			partials = append(partials, partial{
				ofs:   r.ofs + r.len,
				start: true,
				str:   string(b[lastLineEnd+1:]),
			})
		}
		b0 := uintptr(unsafe.Pointer(&b[0]))
		ptr := b0 + uintptr(firstLineEnd) + 1
		end := b0 + uintptr(lastLineEnd) + 1
		for ptr < end {
			hash, val, nameLen, lineLen := parse(ptr)

			ok, e := w.h.find(hash)
			if ok {
				e.count++
				e.sum += val
				e.minVal = min(e.minVal, val)
				e.maxVal = max(e.maxVal, val)
			} else {
				e.hash = hash
				e.name = string(b[ptr-b0 : ptr-b0+nameLen])
				e.count = 1
				e.sum = val
				e.minVal = val
				e.maxVal = val
			}
			ptr += lineLen
		}
	}
	w.partials = partials
	wg.Done()
}

func parse(ptr uintptr) (hash uint64, val int64, nameLen, lineLen uintptr) {
	sep := ptr + 1
	for ; *(*byte)(unsafe.Pointer(sep)) != ';'; sep++ {
	}
	nameLen = sep - ptr

	for ; ptr+8 < sep; ptr += 8 {
		hash ^= *(*uint64)(unsafe.Pointer(ptr))
		hash *= 7
	}
	hash ^= *(*uint64)(unsafe.Pointer(ptr)) & ((1 << ((sep - ptr) * 8)) - 1)

	// Let's try to parse without any conditionals.
	//
	// Four possibilities:
	//
	//   a.b\n         ?? ?? 0A bb 2E aa
	//   ab.c\n        ?? 0A cc 2E bb aa
	//   -a.b\n        ?? 0A bb 2E aa 2D
	//   -ab.c\n       0A cc 2E bb aa 2D

	// ASCII values:
	//  -    0x2D      0b00101101
	//  .    0x2E      0b00101110
	//  \n   0x0A      0b00001010
	//  0-9  0x30-0x39 0b0011....

	// Restrict to the lower 5 bytes.
	x := *(*uint64)(unsafe.Pointer(sep + 1)) & 0xFFFFFFFFFF

	// Digits have the 5th bit (0x10) set to 1. The decimal point
	// can be in byte 1 (0x1000), 2 (0x100000) or 3 (0x10000000).
	n := bits.TrailingZeros64((^x) & 0x10101000)
	// Byte 1: n=12, format is "a.b"
	// Byte 2: n=20, format is "ab.c" or "-a.b"
	// Byte 3: n=28, format is "-ab.c"

	// Byte 0 is either a digit or '-'. Again we can check the 5th bit.
	minus := ((^x) >> 4) & 1        // 0 if no minus, or 1 if minus.
	minusMask := (minus - 1) & 0xFF // 0xFF if no minus, or 0 of minus.
	x = (x & (0xFFFFFFFF00 | minusMask)) << (28 - n)
	valUnsigned := ((x>>8)&0x0F)*100 + ((x>>16)&0x0F)*10 + (x>>32)&0x0F
	val = int64(valUnsigned ^ (-minus) + minus)

	return hash, val, nameLen, nameLen + 4 + uintptr(n)>>3
}

func main() {
	var err error
	file, err = os.Open("measurements.txt")
	checkErr(err)
	info, err := file.Stat()
	checkErr(err)
	size := info.Size()
	n := int((size + chunkSize - 1) / chunkSize)
	ch = make(chan fileRange, n)
	for ofs := int64(0); ofs < size; ofs += chunkSize {
		ch <- fileRange{ofs: ofs, len: min(chunkSize, size-ofs)}
	}
	close(ch)
	numWorkers := runtime.GOMAXPROCS(0)
	wg.Add(numWorkers)
	workers := make([]*worker, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workers[i] = new(worker)
		go workers[i].run()
	}
	wg.Wait()

	// Process partials.
	var partials []partial
	for _, w := range workers {
		partials = append(partials, w.partials...)
	}
	sort.Slice(partials, func(i, j int) bool {
		return partials[i].ofs < partials[j].ofs ||
			(partials[i].ofs == partials[j].ofs && partials[i].start && !partials[j].start)
	})

	buf := make([]byte, 1024)
	ptr := uintptr(unsafe.Pointer(&buf[0]))
	for i := 0; i < len(partials); i++ {
		buf = append(buf[:0], []byte(partials[i].str)...)
		if i+1 < len(partials) && partials[i+1].ofs == partials[i].ofs {
			i++
			buf = append(buf, []byte(partials[i].str)...)
		}
		hash, val, nameLen, _ := parse(ptr)

		ok, e := workers[0].h.find(hash)
		if ok {
			e.count++
			e.sum += val
			e.minVal = min(e.minVal, val)
			e.maxVal = max(e.maxVal, val)
		} else {
			e.hash = hash
			e.name = string(buf[:nameLen])
			e.count = 1
			e.sum = val
			e.minVal = val
			e.maxVal = val
		}
	}

	// Merge hash tables into the first worker's.
	for _, w := range workers[1:] {
		for _, e := range w.h.table {
			if e.hash != 0 {
				ok, e0 := workers[0].h.find(e.hash)
				if ok {
					e0.count += e.count
					e0.sum += e.sum
					e0.minVal = min(e.minVal, e0.minVal)
					e0.maxVal = max(e.maxVal, e0.maxVal)
				} else {
					*e0 = e
				}
			}
		}
	}

	// Generate output.
	res := make([]string, 0, 1024)
	for _, e := range workers[0].h.table {
		if e.hash != 0 {
			res = append(res, fmt.Sprintf("%s=%.1f/%.1f/%.1f",
				e.name,
				float64(e.minVal)*0.1,
				float64(e.sum)/float64(e.count)*0.1,
				float64(e.maxVal)*0.1,
			))
		}
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i] < res[j]
	})
	for _, l := range res {
		fmt.Println(l)
	}
}

func checkErr(err error) {
	if err != nil {
		panic(err)
	}
}

const hashTableSize = 1024 * 512

type hashTable struct {
	table [hashTableSize]hashTableEntry
}

type hashTableEntry struct {
	hash                       uint64
	minVal, maxVal, sum, count int64
	name                       string
}

// find returns the entry containing the given hash, or an empty entry that can
// be used to insert it.
func (h *hashTable) find(hash uint64) (ok bool, _ *hashTableEntry) {
	for i := hash % hashTableSize; ; i = (i + 1) % hashTableSize {
		if h.table[i].hash == hash {
			return true, &h.table[i]
		}
		if h.table[i].hash == 0 {
			return false, &h.table[i]
		}
	}
}
