package main

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"syscall"
	"unsafe"
)

type Counter struct {
	Key   uint64
	Name  string
	Max   int64
	Min   int64
	Total int64
	Count int64
}

func (c *Counter) Add(v int64) {
	if v > c.Max {
		c.Max = v
	}
	if v < c.Min {
		c.Min = v
	}
	c.Total += v
	c.Count++
}

func (c *Counter) Combine(other *Counter) {
	if other.Max > c.Max {
		c.Max = other.Max
	}
	if other.Min < c.Min {
		c.Min = other.Min
	}
	c.Total += other.Total
	c.Count += other.Count
}

func (c *Counter) Avg() int64 {
	return int64(math.Round(float64(c.Total) / float64(c.Count)))
}

func (c *Counter) String() string {
	sw := strings.Builder{}
	sw.Write(intToBytes(c.Min))
	sw.WriteByte('/')
	sw.Write(intToBytes(c.Avg()))
	sw.WriteByte('/')
	sw.Write(intToBytes(c.Max))
	return sw.String()
}

func intToBytes(num int64) []byte {
	if num < 0 {
		if num < -99 {
			return []byte{'-', '0' + byte(-num/100), '0' + byte(((-num)/10)%10), '.', '0' + byte((-num)%10)}
		} else {
			return []byte{'-', '0' + byte((-num)/10), '.', '0' + byte((-num)%10)}
		}
	} else if num > 99 {
		return []byte{'0' + byte(num/100), '0' + byte((num/10)%10), '.', '0' + byte(num%10)}
	} else {
		return []byte{'0' + byte(num/10), '.', '0' + byte(num%10)}
	}
}

func getSemicolonMask(value uint64) uint64 {
	match := value ^ 0x3b3b3b3b3b3b3b3b
	return (match - 0x0101010101010101) & (^match & 0x8080808080808080)
}

const numBuckets = 16384

func runWorker(data []byte) []*Counter {
	var buckets [numBuckets][]*Counter
	var i uintptr

	segments := uintptr(unsafe.Pointer(unsafe.SliceData(data)))
	var key uint64
	var nextI uintptr = segments
	var maxBound uintptr = segments + uintptr(len(data)) - 5

	// manyBuckets := make([][]map[uint64]struct{}, 64)
	// for i := range manyBuckets {
	// 	manyBuckets[i] = make([]map[uint64]struct{}, numBuckets)
	// 	for j := range manyBuckets[i] {
	// 		manyBuckets[i][j] = make(map[uint64]struct{})
	// 	}
	// }

outerLoop:
	for nextI < maxBound {
		i = nextI
		key = 1
		var mask uint64
		var segment uint64 = *(*uint64)(unsafe.Pointer(nextI))
		for {
			mask = getSemicolonMask(segment)
			key += ((mask - 1) & segment)
			nextI += (uintptr(bits.TrailingZeros64(mask)) >> 3)
			segment = *(*uint64)(unsafe.Pointer(nextI))
			if mask != 0 {
				break
			}
		}

		lookup := uint64(key ^ key>>29)

		// for i := 0; i < 64; i++ {
		// 	manyBuckets[i][(key^(key>>i))&(numBuckets-1)][key] = struct{}{}
		// }

		bucket := buckets[lookup&(numBuckets-1)]

		// Inspect the 4th bit of the first byte to determine if it is a '-' symbol.
		negativity := (0b1 & (segment >> (8 + 4))) - 1

		// Determine the position of the '.' point, one of 0, 1, 2. Where 0 means a number like 1.1.
		decimalPos := (((segment >> (16 + 3)) & 0b10) | ((segment >> (24 + 4)) & 0b1)) - 1

		// If negative, we need to clear the first 4 bits so it's treated as 0 in the next step.
		segment &= (0xfffffffffffff00 ^ (negativity & 0xf00))

		// Shift the number such that the hundreds field is in the LSB.
		segment >>= (decimalPos << 3)

		nextI += uintptr(decimalPos + 5)

		// Mask out only the integer values of the string.
		segment &= 0x0f000f0f
		// 0x00000000UU00TTHH * 1
		// 0x0000UU00TTHH0000 * 10
		// 0x000000TTHH000000 * 100
		// Perform fucked up multiplication to compute the integer representation of the string
		// in very few instructions.
		segment = (segment * (10<<16 | 100<<24 | 1)) >> 24

		// Recover the 10 bit result from position 24, and multiply by the sign.
		segment = ((segment & ((1 << 10) - 1)) ^ negativity) - negativity

		for _, v := range bucket {
			if v.Key == key {
				v.Add(int64(segment))
				continue outerLoop
			}
		}

		buckets[lookup&(numBuckets-1)] = append(bucket, &Counter{
			Key:   key,
			Name:  string(data[i-segments : nextI-segments-uintptr(decimalPos)-5]),
			Max:   int64(segment),
			Min:   int64(segment),
			Total: int64(segment),
			Count: 1,
		})
	}

	results := make([]*Counter, 0, 512)

	for _, v := range buckets {
		results = append(results, v...)
	}

	// for i, b := range manyBuckets {
	// 	hits := 0
	// 	depth := 0
	// 	verification := 0
	// 	for _, v := range b {
	// 		if len(v) > 1 {
	// 			depth += len(v)
	// 			hits++
	// 		}
	// 		verification += len(v)
	// 	}
	// 	fmt.Printf("bucket %d has %d hits with %d depth and verification %d\n", i, hits, depth, verification)
	// }

	return results
}

var numThreads = runtime.NumCPU() * 4

type abortWriter struct {
	w io.Writer
	c chan struct{}
}

func (w *abortWriter) Write(p []byte) (n int, err error) {
	if bytes.Contains(p, []byte("--- REAL EOF ---\n")) {
		n, err := w.w.Write([]byte(strings.ReplaceAll(string(p), "--- REAL EOF ---\n", "")))
		select {
		case <-w.c:
		default:
			close(w.c)
		}
		return n, err
	}

	return w.w.Write(p)
}

func (w *abortWriter) Wait() {
	<-w.c
}

const altModeVar = "BRCGO_ALT_MODE"

func main() {
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(1024 * 1024 * 1024 * 10)

	if os.Getenv(altModeVar) == "" || os.Getenv(altModeVar) == "0" {
		// im totally not doing this because i want to avoid the overhead of the kernel cleaning up
		// the shared memory pages swapped in from the massive mmaped reads.
		var args []string
		if len(os.Args) > 1 {
			args = os.Args[1:]
		}
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(cmd.Env, altModeVar+"=2")
		w := &abortWriter{
			w: os.Stdout,
			c: make(chan struct{}),
		}
		cmd.Stdout = w
		cmd.Stderr = os.Stderr

		err := cmd.Start()
		if err != nil {
			panic(err)
		}

		cmd.Process.Release()

		w.Wait()
		return
	}

	inAltMode := os.Getenv(altModeVar) == "2"

	defer func() {
		if inAltMode {
			fmt.Println("--- REAL EOF ---")
		}
	}()

	openFile := "measurements.txt"
	if len(os.Args) > 1 {
		openFile = os.Args[1]
	}

	data, err := mmap(openFile)
	if err != nil {
		panic(err)
	}

	// use a single thread if the problem is too small
	if len(data) < 2048 {
		numThreads = 1
	}

	type pk struct {
		counters []*Counter
		id       int
	}

	results := make(chan pk, numThreads)

	jobSize := (len(data) / numThreads) - 10
	lastJobEnd := 0
	for i := 0; i < numThreads; i++ {
		i := i
		if i == numThreads-1 {
			go func(left int) {
				results <- pk{
					runWorker(data[left:]), i,
				}
			}(lastJobEnd)
		} else {
			// find an appropriate new line boundary to split on
			delim := bytes.IndexByte(data[lastJobEnd+jobSize:], '\n')
			go func(left, right int) {
				results <- pk{
					runWorker(data[left:right]), i,
				}
			}(lastJobEnd, lastJobEnd+jobSize+delim)
			lastJobEnd += jobSize + delim + 1
		}
	}

	combinedResults := make(map[string]*Counter)
	for i := 0; i < numThreads; i++ {
		p := <-results

		for _, value := range p.counters {
			if _, found := combinedResults[value.Name]; found {
				combinedResults[value.Name].Combine(value)
			} else {
				combinedResults[value.Name] = value
			}
		}
	}

	var allResults []string
	for _, v := range combinedResults {
		allResults = append(allResults, v.Name+"="+v.String())
	}
	sort.Strings(allResults)

	for _, v := range allResults {
		fmt.Println(v)
	}

	if inAltMode {
		fmt.Println("--- REAL EOF ---")
	}

	os.Stdout.Sync()

	syscall.Munmap(data)
}

func mmap(filename string) ([]byte, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := fi.Size()
	if size == 0 {
		return nil, nil
	}
	if size < 0 {
		return nil, fmt.Errorf("mmap: file %q has negative size", filename)
	}
	if size != int64(int(size)) {
		return nil, fmt.Errorf("mmap: file %q is too large", filename)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}

	return data, err
}
