package main

import (
	"bytes"
	"fmt"
	"log"
	"math"
	"math/bits"
	"os"
	"sort"
	"strconv"
	"sync"
	"syscall"
	"unsafe"
)

type hashtable [1 << 16]*PartitionResult

// Given memory-mapped data in [baseAddr, endAddr], compute consecutive partitions of a fixed size, and append them to the workqueue.
// Each partition is represented as a pair of [start, end) addresses.
func producePartitions(baseAddr, endAddr int64, workqueue chan<- [2]int64) {
	scanner := newScanner(baseAddr, endAddr)
	// Empirically chosen partition size of 32MB seems to hit the sweet spot wrt maximum throughput.
	// Roughly, this corresponds to ~40ms per partition. Thus, 32MB * 10 workers/40ms gets us 8GB/s.
	partSize := int64(1 << 25)

	for i := baseAddr; i <= endAddr; {
		partEndAddr := i + partSize
		// advance to the start of the next input row
		for ; partEndAddr < endAddr && (scanner.nextAt(partEndAddr)&0xFF) != '\n'; partEndAddr++ {
			// nop
		}
		// skip over the newline
		partEndAddr++
		partEndAddr = min(partEndAddr, endAddr)

		if i == partEndAddr {
			break
		}
		workqueue <- [2]int64{i, partEndAddr}
		i += (partEndAddr - i)
	}
}

//gcassert:inline
func findNewline(word int64) int {
	// N.B. 0x3B is the ASCII code for ';'. Thus, the mask sets every semicolon occurrence to 0x00.
	maskedInput := uint64(word ^ 0x0A0A0A0A0A0A0A0A)
	// Transform each 0x00 into 0x80, everything else into 0x00.
	// N.B. This is a classic problem of finding the _rightmost_ zero byte. (Recall, we're in little-endian.)
	// N.B. Left conjunct ensures 0x00 turns into 0xFF. If some non-zero byte has a 0 in MSB, it gets cleared by left
	// conjunct; otherwise, 0x80 clears the rest.
	tmp := (maskedInput - 0x0101010101010101) & ^maskedInput & 0x8080808080808080
	// Return the index of the rightmost zero byte or 8, if there are no zero bytes.
	return bits.LeadingZeros64(tmp) >> 3
}

// This is an example of premature optimization :)
//
// runtime.NumCPU() is _technically_ not without overhead, but its value is populated unconditionally at startup.
// As can be seen from this ARM64 assembly, `osinit` populates `runtime.ncpu` with the number of CPUs.
//
// MOVW    8(RSP), R0      // copy argc
// MOVW    R0, -8(RSP)
// MOVD    16(RSP), R0     // copy argv
// MOVD    R0, 0(RSP)
// BL      runtime·args(SB)
// BL      runtime·osinit(SB)
// BL      runtime·schedinit(SB)
//
// // create a new goroutine to start program
// MOVD    $runtime·mainPC(SB), R0         // entry
// ...
const numWorkers = 10

func main() {
	data := mmap("measurements.txt")
	baseAddr := (int64)(uintptr(unsafe.Pointer(&data.d[0])))
	endAddr := (int64)(uintptr(unsafe.Pointer(&data.d[len(data.d)-1])))

	// Buffer the workqueue since the producer isn't computing much.
	workqueue := make(chan [2]int64, numWorkers*1000)
	donequeue := make(chan int, numWorkers)

	results := make([]hashtable, numWorkers)
	// spawn goroutines and wait for them to finish
	// escapes to heap, see https://github.com/golang/go/issues/33216
	var wg sync.WaitGroup
	//
	mergedPart1 := []*AggResult{}
	mergedPart2 := []*AggResult{}
	//start := time.Now()

	wg.Add(1)
	go func() {
		defer wg.Done()

		producePartitions(baseAddr, endAddr, workqueue)
		close(workqueue)
		//fmt.Printf("done producing partitions: %v\n", time.Since(start))

		finishedBitset := uint64(0)
		mergedBitset := uint64(0)

		for {
			workerId := <-donequeue
			// set workerId-th bit to 1
			finishedBitset |= 1 << workerId

			if bits.OnesCount64(finishedBitset) == numWorkers {
				// We're done. The main goroutine will finish the job.

				// subtract mergedBitset from finishedBitset
				finishedBitset &= ^mergedBitset
				mergedPart2 = sortMerge(results, bitsetToSlice(finishedBitset))

				break
			}
			if bits.OnesCount64(finishedBitset) >= 5 && mergedBitset == 0 {
				mergedPart1 = sortMerge(results, bitsetToSlice(finishedBitset))
				mergedBitset = finishedBitset
			}
		}
		// Clean up.
		close(donequeue)
	}()

	for i := 0; i < numWorkers; i++ {
		// N.B. Heavy-lifting is done by the these workers.
		wg.Add(1)
		go func(workerId int) {
			defer func() {
				donequeue <- workerId
				wg.Done()
			}()

			// N.B. This stack-allocated hashtable is reused all partitions processed by this worker.
			//gcassert:noescape
			ht := hashtable{}

			//fmt.Printf("workerId: %d is ready; time elapsed: %v\n", workerId, time.Since(start))

			//workerProcessStart := time.Now()

			for partition := range workqueue {
				//runtime.LockOSThread()
				//fmt.Printf("workerId: %d, partition: %v\n", workerId, partition)
				//workerPartStart := time.Now()
				processPartition(baseAddr, partition[0], partition[1], &ht)
				//fmt.Printf("workerId: %d partition[%d, %d) processing time: %v\n", workerId, partition[0], partition[1], time.Since(workerPartStart))

			}
			results[workerId] = ht
			//fmt.Printf("workerId: %d total processing time: %v\n", workerId, time.Since(workerProcessStart))
		}(i)

	}

	wg.Wait()

	// Multi-merge appears to be slightly faster.
	mergeAndPrint(mergedPart1, mergedPart2)

	//singlePassMergeAndPrint(results)
}

//gcassert:inline
func bitsetToSlice(bitset uint64) []int {
	res := []int{}
	for i := 0; i < numWorkers; i++ {
		if (bitset>>i)&1 == 1 {
			res = append(res, i)
		}
	}
	return res
}

//func print(a []*AggResult) {
//	for _, entry := range a {
//		fmt.Printf("%s=%s/%s/%s\n", entry.name,
//			formatFloat(round(float64(entry.min)/float64(10.0))),
//			formatFloat(round(float64(entry.sum)/float64(10.0)/float64(entry.count))),
//			formatFloat(round(float64(entry.max)/float64(10.0))))
//	}
//}

func singlePassMergeAndPrint(results []hashtable) {
	// copy results
	// N.B. anything above 1<<14 escapes to heap due to size _and_ interface conversion when invoking sort.Slice!
	// TODO: in case results are larger, we'll need to restart using heap allocation
	// see https://stackoverflow.com/a/69187698
	//
	tmp := []*AggResult{}
	for _, workerRes := range results {
		for _, entry := range workerRes {
			if entry != nil {
				tmp = append(tmp, &AggResult{
					name:  string(unsafe.Slice((*byte)(unsafe.Pointer(uintptr(entry.namePos))), entry.nameLength)),
					min:   entry.min,
					max:   entry.max,
					sum:   entry.sum,
					count: entry.count,
				})
			}
		}
	}
	// tmp escapes to heap owing to https://github.com/golang/go/issues/17332
	// TODO: try inlining.
	sort.Slice(tmp, func(i, j int) bool {
		return tmp[i].name < tmp[j].name
	})

	i := 1
	for ; i < len(tmp); i++ {
		if tmp[i].name == tmp[i-1].name {
			merge(tmp[i-1], tmp[i])
		} else {
			entry := tmp[i-1]
			fmt.Printf("%s=%s/%s/%s\n", entry.name,
				formatFloat(round(float64(entry.min)/float64(10.0))),
				formatFloat(round(float64(entry.sum)/float64(10.0)/float64(entry.count))),
				formatFloat(round(float64(entry.max)/float64(10.0))))
		}
	}
	entry := tmp[i-1]
	fmt.Printf("%s=%s/%s/%s\n", entry.name,
		formatFloat(round(float64(entry.min)/float64(10.0))),
		formatFloat(round(float64(entry.sum)/float64(10.0)/float64(entry.count))),
		formatFloat(round(float64(entry.max)/float64(10.0))))
}

func mergeAndPrint(a []*AggResult, b []*AggResult) {
	i := 0
	j := 0

	for i < len(a) && j < len(b) {
		var entry *AggResult
		if a[i].name == b[j].name {
			merge(a[i], b[j])
			// N.B. merge result is in the second arg.
			entry = b[j]
			i++
			j++
		} else if a[i].name < b[j].name {
			entry = a[i]
			i++
		} else {
			entry = b[j]
			j++
		}
		fmt.Printf("%s=%s/%s/%s\n", entry.name,
			formatFloat(round(float64(entry.min)/float64(10.0))),
			formatFloat(round(float64(entry.sum)/float64(10.0)/float64(entry.count))),
			formatFloat(round(float64(entry.max)/float64(10.0))))
	}
	for ; i < len(a); i++ {
		entry := a[i]
		fmt.Printf("%s=%s/%s/%s\n", entry.name,
			formatFloat(round(float64(entry.min)/float64(10.0))),
			formatFloat(round(float64(entry.sum)/float64(10.0)/float64(entry.count))),
			formatFloat(round(float64(entry.max)/float64(10.0))))
	}
	for ; j < len(b); j++ {
		entry := b[j]
		fmt.Printf("%s=%s/%s/%s\n", entry.name,
			formatFloat(round(float64(entry.min)/float64(10.0))),
			formatFloat(round(float64(entry.sum)/float64(10.0)/float64(entry.count))),
			formatFloat(round(float64(entry.max)/float64(10.0))))
	}
}

func sortMerge(parts []hashtable, finishedWorkers []int) []*AggResult {
	tmp := []*AggResult{}
	for _, workerId := range finishedWorkers {
		workerRes := parts[workerId]
		n := 0
		for _, entry := range workerRes {
			if entry != nil {
				tmp = append(tmp, &AggResult{
					name:  string(unsafe.Slice((*byte)(unsafe.Pointer(uintptr(entry.namePos))), entry.nameLength-1)),
					min:   entry.min,
					max:   entry.max,
					sum:   entry.sum,
					count: entry.count,
				})
				n++
			}
		}
	}

	// tmp escapes to heap owing to https://github.com/golang/go/issues/17332
	// TODO: try inlining.
	sort.Slice(tmp, func(i, j int) bool {
		return tmp[i].name < tmp[j].name
	})

	i := 1
	j := 0
	for ; i < len(tmp); i++ {
		if tmp[i].name == tmp[i-1].name {
			merge(tmp[i-1], tmp[i])
		} else {
			entry := tmp[i-1]
			tmp[j] = entry
			j++
		}
	}
	entry := tmp[i-1]
	tmp[j] = entry

	return tmp[:j+1]
}

func processPartition(base, start, end int64, ht *hashtable) {
	var entry *PartitionResult
	var semiclnWordBytePos int
	var semiclnPos int64
	var semiclnWord int64

	scanner := newScanner(start, end)
	// Process each input row.
	for scanner.hasNext() {
		lineStart := scanner.pos
		hash := int64(0)
		word := scanner.next()
		firstWord := word
		prevWord := int64(0)
		prevWordPos := int64(0)
		// read until newline
		i := findNewline(word)

		for ; i == 8; i = findNewline(word) {
			prevWordPos = scanner.pos
			prevWord = word
			hash ^= prevWord
			scanner.advance(8)
			word = scanner.next()
		}
		wordPos := scanner.pos
		semiclnWordBytePos = findSemicolon(prevWord)
		semiclnPos = prevWordPos + int64(7-semiclnWordBytePos)
		semiclnWord = prevWord
		// clear hash for prevWord
		hash ^= prevWord

		if semiclnWordBytePos == 8 {
			semiclnWordBytePos = findSemicolon(word)
			semiclnPos = wordPos + int64(7-semiclnWordBytePos)
			semiclnWord = word
			// add hash for prevWord
			hash ^= prevWord
		}
		lastNamePart := maskWord(semiclnWord, semiclnWordBytePos)
		hash ^= lastNamePart
		// N.B. station name is in [lineStart, nlPos - 8 + semicolonPos]
		stationNameLen := int((semiclnPos - lineStart + 1))
		// N.B. this is potentially an _unaligned_ load
		// Technically, it's possible but cumbersome to mask out the bits from prevWord and word.
		alignedWord := scanner.nextAt(semiclnPos + 1)
		nlWordBitPos := int64(7 - i)
		decmlPos := wordPos + nlWordBitPos - 2
		decmlWordBitPos := int(((decmlPos - semiclnPos - 1) << 3) + 4)
		// N.B. instead of searching for the decimal separator, we can simply compute its position per above.
		//decmlWordBitPos := findDecimalSeparator(alignedWord)
		temperature := convertIntoNumber(decmlWordBitPos, alignedWord)
		// update hashtable
	outer:
		for {
			index := hashToIndex(hash, len(ht))
			entry = ht[index] //gcassert:bce

			if entry == nil {
				if stationNameLen <= 8 {
					// Make sure that firstWord is properly masked.
					firstWord = lastNamePart
				}
				ht[index] = newEntry(lineStart, stationNameLen, temperature, firstWord, lastNamePart) //gcassert:bce
				break
			}
			// Check for collision.
			if stationNameLen <= 8 {
				if entry.firstNamePart != lastNamePart {
					index = (index + 31) & (len(ht) - 1)
					continue outer
				}
			} else if stationNameLen <= 16 {
				if !(entry.firstNamePart == firstWord && entry.lastNamePart == lastNamePart) {
					index = (index + 31) & (len(ht) - 1)
					continue outer
				}
			} else {
				// Slow case.
				entryNameBytes := unsafe.Slice((*byte)(unsafe.Pointer(uintptr(entry.namePos))), stationNameLen)
				nameBytes := unsafe.Slice((*byte)(unsafe.Pointer(uintptr(lineStart))), stationNameLen)

				if !bytes.Equal(entryNameBytes, nameBytes) {
					index = (index + 31) & (len(ht) - 1)
					continue outer
				}
			}
			updateEntry(entry, temperature)
			break
		}
		// advance to the next input row
		scanner.advance(nlWordBitPos + 1)
	}
}

// Returns the byte index ([0, 7]) of the rightmost semicolon occurrence in the given word.
// If the word does not contain a semicolon, returns 8.
// N.B. Owing to little-endian, the rightmost byte is the _first_ byte (i.e., index 0).
//
//gcassert:inline
func findSemicolon(word int64) int {
	// N.B. 0x3B is the ASCII code for ';'. Thus, the mask sets every semicolon occurrence to 0x00.
	maskedInput := uint64(word ^ 0x3B3B3B3B3B3B3B3B)
	// Transform each 0x00 into 0x80, everything else into 0x00.
	// N.B. This is a classic problem of finding the _rightmost_ zero byte. (Recall, we're in little-endian.)
	// N.B. Left conjunct ensures 0x00 turns into 0xFF. If some non-zero byte has a 0 in MSB, it gets cleared by left
	// conjunct; otherwise, 0x80 clears the rest.
	tmp := (maskedInput - 0x0101010101010101) & ^maskedInput & 0x8080808080808080
	// Return the index of the rightmost zero byte or 8, if there are no zero bytes.
	return bits.LeadingZeros64(tmp) >> 3
}

//gcassert:inline
func findDecimalSeparator(word int64) int {
	// N.B. To find the decimal separator, we follow the same approach as in findSemicolon.
	// However, observe the following pattern:
	// The 4th binary digit of the ascii of a _digit_ is 1, e.g., 0x31 = 0b0011|0001,
	// that of the '.' is 0, i.e, 0x2e = 0b0010|1110.
	// Furthermore, the decimal separator can occur only in positions 12, 20, 28, corresponding to
	// whether temperature is negative, and contains one (or two) decimals before the separator.
	// Thus, the following mask suffices to clear all decimals and keep the decimal separator bit.
	return bits.TrailingZeros64(uint64(^word & 0x10101000))
}

//gcassert:inline
func convertIntoNumber(decimalSepPos int, word int64) int16 {
	shift := 28 - decimalSepPos
	// signed is -1 if negative, 0 otherwise
	signed := (^word << 59) >> 63
	designMask := ^(signed & 0xFF)
	// Align the number to a specific position and transform the ascii code
	// to actual digit value in each byte
	digits := ((word & designMask) << shift) & int64(0x0F000F0F00)

	// Now digits is in the form 0xUU00TTHH00 (UU: units digit, TT: tens digit, HH: hundreds digit)
	// 0xUU00TTHH00 * (100 * 0x1000000 + 10 * 0x10000 + 1) =
	// 0x000000UU00TTHH00 +
	// 0x00UU00TTHH000000 * 10 +
	// 0xUU00TTHH00000000 * 100
	// Now TT * 100 has 2 trailing zeroes and HH * 100 + TT * 10 + UU < 0x400
	// This results in our value lies in the bit 32 to 41 of this product
	// That was close :)
	absValue := ((digits * 0x640a0001) >> 32) & 0x3FF
	value := (absValue ^ signed) - signed
	return int16(value)
}

//gcassert:inline
func maskWord(word int64, semiclnWordBytePos int) int64 {
	// Shift 1s into [0, semiclnWordBytePos] to form the mask.
	return word & int64(^uint(0)>>(semiclnWordBytePos<<3))
}

// Returns the index of the hash table entry for the given (key) hash.
// requires: sizePowerOfTwo is a power of two
//
//gcassert:inline
func hashToIndex(hash int64, sizePowerOfTwo int) int {
	hash32 := int(hash ^ (hash >> 28))
	finalHash := hash32 ^ (hash32 >> 15)
	// N.B. sizePowerOfTwo is a power of two, so the following is equivalent to finalHash % sizePowerOfTwo.
	return finalHash & (sizePowerOfTwo - 1)
}

type PartitionResult struct {
	nameLength                  int
	min, max                    int16
	firstNamePart, lastNamePart int64
	namePos                     int64
	count, sum                  int64
}

type AggResult struct {
	min, max   int16
	count, sum int64
	name       string
}

//gcassert:inline
func merge(a *AggResult, b *AggResult) {
	b.count += a.count
	b.sum += a.sum
	b.min = min(a.min, b.min)
	b.max = max(a.max, b.max)
}

//gcassert:inline
func newEntry(inputRowPos int64, stationNameLen int, temperature int16, firstNamePart, lastNamePart int64) *PartitionResult {
	entry := &PartitionResult{
		namePos:       inputRowPos,
		nameLength:    stationNameLen,
		min:           temperature,
		max:           temperature,
		count:         1,
		sum:           int64(temperature),
		firstNamePart: firstNamePart,
		lastNamePart:  lastNamePart,
	}
	return entry
}

//gcassert:inline
func updateEntry(entry *PartitionResult, number int16) {
	entry.sum += int64(number)
	if number < entry.min {
		entry.min = number
		return
	}
	if number > entry.max {
		entry.max = number
	}
	entry.count++
}

// ==================================================Helpers===================================================
type scanner struct {
	pos, end int64
}

//gcassert:inline
func newScanner(start, end int64) *scanner {
	return &scanner{pos: start, end: end}
}

//gcassert:inline
func (s *scanner) next() int64 {
	return *(*int64)(unsafe.Pointer(uintptr(s.pos)))
}

//gcassert:inline
func (s *scanner) nextByte() byte {
	return *(*byte)(unsafe.Pointer(uintptr(s.pos)))
}

//gcassert:inline
func (s *scanner) nextAt(pos int64) int64 {
	return *(*int64)(unsafe.Pointer(uintptr(pos)))
}

//gcassert:inline
func (s *scanner) nextByteAt(pos int64) byte {
	return *(*byte)(unsafe.Pointer(uintptr(pos)))
}

//gcassert:inline
func (s *scanner) hasNext() bool {
	return s.pos < s.end
}

//gcassert:inline
func (s *scanner) advance(delta int64) {
	s.pos += delta
}

//gcassert:inline
func (s *scanner) setPos(l int64) {
	s.pos = l
}

// N.B. There are obviously faster ways of formatting fixed-point numbers. For a short survey, see
// https://lemire.me/blog/2021/11/18/converting-integers-to-fix-digit-representations-quickly/
//
//gcassert:inline
func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 1, 64)
}

//gcassert:inline
func round(value float64) float64 {
	return math.Round(value*10.0) / 10.0
}

//==================================================mmap from Russ Cox=================================================
// Lifted from: https://github.com/google/codesearch/blob/8ba29bd255b740aee4eb4e4ddb5d7ec0b4d9f23e/index/mmap_linux.go#L13

// An mmapData is mmap'ed read-only data from a file.
type mmapData struct {
	f *os.File
	d []byte
}

// mmap maps the given file into memory.
func mmap(file string) mmapData {
	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	return mmapFile(f)
}

func mmapFile(f *os.File) mmapData {
	st, err := f.Stat()
	if err != nil {
		log.Fatal(err)
	}
	size := st.Size()
	if int64(int(size+4095)) != size+4095 {
		log.Fatalf("%s: too large for mmap", f.Name())
	}
	n := int(size)
	if n == 0 {
		return mmapData{f, nil}
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, (n+4095)&^4095, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		log.Fatalf("mmap %s: %v", f.Name(), err)
	}
	return mmapData{f, data[:n]}
}

//==================================================Appendix=============

// SWAR arithmetic [1] turns out to be _slower_ than plain addition of 64-bit registers.
// [1] https://www.chessprogramming.org/SIMD_and_SWAR_Techniques
//
//gcassert:inline
func swarAdd(x, acc uint64) uint64 {
	return ((x & ^uint64(0x8000800080008000)) + (acc & ^uint64(0x8000800080008000))) ^ ((x ^ acc) & uint64(0x8000800080008000))
}

// Branchless min.
//
//gcassert:inline
func min_(x, y int64) int64 {
	return y ^ ((x ^ y) & ((x - y) >> 63))
}

// Branchless max.
//
//gcassert:inline
func max_(x, y int64) int64 {
	return x ^ ((x ^ y) & ((x - y) >> 63))
}
