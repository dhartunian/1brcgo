// 2024, michae2

package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/maphash"
	"math"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"syscall"
	"unsafe"
)

const (
	tableN    = 1 << 11
	tableMask = 1<<11 - 1
	chunk     = 1 << 26
	maxProbe  = 1 << 5
)

type record struct {
	key []byte
	sum int64
	cnt uint32
	min int16
	max int16
}

func (r *record) getKey() []byte {
	key := r.key
	keylen := len(key)
	if keylen <= 8 {
		header := (*reflect.SliceHeader)(unsafe.Pointer(&r.key))
		key = binary.LittleEndian.AppendUint64(nil, uint64(header.Data))
		key = key[:keylen]
	} else if keylen < 16 {
		header := (*reflect.SliceHeader)(unsafe.Pointer(&r.key))
		key = binary.LittleEndian.AppendUint64(nil, uint64(header.Data))
		key = binary.LittleEndian.AppendUint64(key, uint64(header.Cap))
		key = key[:keylen]
	}
	return key
}

func (r *record) String() string {
	return fmt.Sprintf(
		"%s=%.1f/%.1f/%.1f\n",
		r.getKey(), float64(r.min)/10.0, float64(r.sum)/float64(r.cnt)/10.0, float64(r.max)/10.0,
	)
}

type table []record

func main() {
	//syscall.Setpriority(syscall.PRIO_PROCESS, 0, -20)

	// disable GC
	debug.SetMemoryLimit(math.MaxInt64)
	debug.SetGCPercent(-1)

	/*
		pfile, err := os.Create("cpu.prof")
		if err != nil {
			panic(err)
		}
		defer pfile.Close()
		pprof.StartCPUProfile(pfile)
		defer pprof.StopCPUProfile()
	*/

	// mmap the measurements file
	file, err := os.Open("measurements.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	fd := int(file.Fd())

	fileInfo, err := file.Stat()
	if err != nil {
		panic(err)
	}
	fileSize := int(fileInfo.Size())

	data, err := syscall.Mmap(fd, 0, fileSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		panic(err)
	}
	defer syscall.Munmap(data)

	workers := runtime.NumCPU()

	// build a queue of file chunks
	chunks := make(chan []byte, len(data)/chunk+1+workers)
	start := 0
	for i := chunk; i < len(data); i += chunk {
		end := i - 1
		for data[end] != '\n' {
			end++
		}
		chunks <- data[start : end+1]
		start = end + 1
	}
	chunks <- data[start:]
	for i := 0; i < workers; i++ {
		chunks <- nil
	}

	// run workers
	seed := maphash.MakeSeed()
	var resultsMu sync.Mutex
	var results [tableN]record
	var wait sync.WaitGroup
	for i := 0; i < workers; i++ {
		wait.Add(1)
		go parse(chunks, seed, &resultsMu, results[:], &wait)
	}
	wait.Wait()

	// sort and print the aggregated results
	sortedResults := make([]string, 0, 600)
	for i := range results {
		rec := &results[i]
		if rec.cnt == 0 {
			continue
		}
		sortedResults = append(sortedResults, rec.String())
	}
	sort.Strings(sortedResults)
	for _, res := range sortedResults {
		fmt.Print(res)
	}
}

func parse(
	chunks <-chan []byte,
	seed maphash.Seed,
	resultsMu *sync.Mutex,
	results table,
	wait *sync.WaitGroup,
) {
	ht := make(table, tableN)
	for subdata := range chunks {
		if subdata == nil {
			// no more chunks
			// merge our hash table into the results hash table
			resultsMu.Lock()
		next_bucket:
			for i := range ht {
				rec := &ht[i]
				if rec.cnt == 0 {
					continue
				}
				hash := maphash.Bytes(seed, rec.getKey())
				for j := uint64(0); j < maxProbe; j++ {
					// linear probing
					resultRec := &results[int((hash+j)&tableMask)]
					if resultRec.cnt == 0 {
						*resultRec = *rec
						continue next_bucket
					} else if len(rec.key) != len(resultRec.key) {
						continue
					} else if len(rec.key) < 16 {
						recHeader := (*reflect.SliceHeader)(unsafe.Pointer(&rec.key))
						resHeader := (*reflect.SliceHeader)(unsafe.Pointer(&resultRec.key))
						if *resHeader != *recHeader {
							continue
						}
					} else if !bytes.Equal(resultRec.key, rec.key) {
						continue
					}
					// merge the record
					resultRec.sum += rec.sum
					resultRec.cnt += rec.cnt
					if rec.min < resultRec.min {
						resultRec.min = rec.min
					}
					if rec.max > resultRec.max {
						resultRec.max = rec.max
					}
					continue next_bucket
				}
				// TODO: grow hash table
				panic("too many probes")
			}
			resultsMu.Unlock()
			wait.Done()
			return
		}

	next_line:
		for off := 0; off < len(subdata); {

			start := off

			// find the separator
			// this is the most expensive part of the whole program!

			// use some bit twiddling to check the first 8 bytes
			var inlineA, inlineB uint64
			if off+8 < len(subdata) {
				inlineA = *(*uint64)(unsafe.Pointer(&subdata[off]))
				// bit twiddling trick to find byte(s) matching 0x3b
				found := (inlineA ^ 0x3b3b3b3b3b3b3b3b - 0x0101010101010101) & ^inlineA & 0x8080808080808080
				switch found {
				case 0x8000000000008000:
					fallthrough
				case 0x0000000000008000:
					inlineA &= 0x00000000000000ff
					off += 1
					goto found
				case 0x0000000000800000:
					inlineA &= 0x000000000000ffff
					off += 2
					goto found
				case 0x0000000080000000:
					inlineA &= 0x0000000000ffffff
					off += 3
					goto found
				case 0x0000008000000000:
					inlineA &= 0x00000000ffffffff
					off += 4
					goto found
				case 0x0000800000000000:
					inlineA &= 0x000000ffffffffff
					off += 5
					goto found
				case 0x0080000000000000:
					inlineA &= 0x0000ffffffffffff
					off += 6
					goto found
				case 0x8000000000000000:
					inlineA &= 0x00ffffffffffffff
					off += 7
					goto found
				}
			}

			// again some bit twiddling to check the next 8 bytes
			if off+16 < len(subdata) {
				inlineB = *(*uint64)(unsafe.Pointer(&subdata[off+8]))
				// find byte(s) matching 0x3b
				found := (inlineB ^ 0x3b3b3b3b3b3b3b3b - 0x0101010101010101) & ^inlineB & 0x8080808080808080
				switch found {
				case 0x0080000000000080:
					fallthrough
				case 0x8000000000000080:
					fallthrough
				case 0x0000000000000080:
					inlineB = 0
					off += 8
					goto found
				case 0x8000000000008000:
					fallthrough
				case 0x0000000000008000:
					inlineB &= 0x00000000000000ff
					off += 9
					goto found
				case 0x0000000000800000:
					inlineB &= 0x000000000000ffff
					off += 10
					goto found
				case 0x0000000080000000:
					inlineB &= 0x0000000000ffffff
					off += 11
					goto found
				case 0x0000008000000000:
					inlineB &= 0x00000000ffffffff
					off += 12
					goto found
				case 0x0000800000000000:
					inlineB &= 0x000000ffffffffff
					off += 13
					goto found
				case 0x0080000000000000:
					inlineB &= 0x0000ffffffffffff
					off += 14
					goto found
				case 0x8000000000000000:
					inlineB &= 0x00ffffffffffffff
					off += 15
					goto found
				}
			}

			// fall back to the slower way
			off += bytes.IndexByte(subdata[off:], ';')

			// might have to build our inlines anyway
			if off-start < 16 {
				inlineA = 0
				inlineB = 0
				switch off - start {
				case 15:
					inlineB += uint64(subdata[start+14]) << 48
					fallthrough
				case 14:
					inlineB += uint64(subdata[start+13]) << 40
					fallthrough
				case 13:
					inlineB += uint64(subdata[start+12]) << 32
					fallthrough
				case 12:
					inlineB += uint64(subdata[start+11]) << 24
					fallthrough
				case 11:
					inlineB += uint64(subdata[start+10]) << 16
					fallthrough
				case 10:
					inlineB += uint64(subdata[start+9]) << 8
					fallthrough
				case 9:
					inlineB += uint64(subdata[start+8])
					fallthrough
				case 8:
					inlineA += uint64(subdata[start+7]) << 56
					fallthrough
				case 7:
					inlineA += uint64(subdata[start+6]) << 48
					fallthrough
				case 6:
					inlineA += uint64(subdata[start+5]) << 40
					fallthrough
				case 5:
					inlineA += uint64(subdata[start+4]) << 32
					fallthrough
				case 4:
					inlineA += uint64(subdata[start+3]) << 24
					fallthrough
				case 3:
					inlineA += uint64(subdata[start+2]) << 16
					fallthrough
				case 2:
					inlineA += uint64(subdata[start+1]) << 8
					fallthrough
				case 1:
					inlineA += uint64(subdata[start])
				}
			}

		found:
			key := subdata[start:off]

			// hash the key
			hash := maphash.Bytes(seed, key)

			if len(key) < 16 {
				// inline the key
				header := (*reflect.SliceHeader)(unsafe.Pointer(&key))
				header.Data = uintptr(inlineA)
				header.Cap = int(inlineB)
			}

			// skip over the ;
			off++

			// parse the temperature
			sign := 1
			if subdata[off] == '-' {
				sign = -1
				off++
			}

			temp := int(subdata[off] - '0')
			off++

			if subdata[off] == '.' {
				off++
			} else {
				temp *= 10
				temp += int(subdata[off] - '0')
				// skip over the .
				off += 2
			}

			temp *= 10
			temp += int(subdata[off] - '0')
			temp *= sign
			// skip over the newline
			off += 2

			// find the record
			for i := uint64(0); i < maxProbe; i++ {
				// linear probing
				rec := &ht[int((hash+i)&tableMask)]
				if rec.cnt == 0 {
					rec.key = key
					rec.sum = int64(temp)
					rec.cnt = 1
					rec.min = int16(temp)
					rec.max = int16(temp)
					continue next_line
				} else if len(key) != len(rec.key) {
					continue
				} else if len(key) < 16 {
					keyHeader := (*reflect.SliceHeader)(unsafe.Pointer(&key))
					recHeader := (*reflect.SliceHeader)(unsafe.Pointer(&rec.key))
					if *recHeader != *keyHeader {
						continue
					}
				} else if !bytes.Equal(rec.key, key) {
					continue
				}
				// update the record
				rec.sum += int64(temp)
				rec.cnt++
				if int16(temp) < rec.min {
					rec.min = int16(temp)
				}
				if int16(temp) > rec.max {
					rec.max = int16(temp)
				}
				continue next_line
			}
			// TODO: grow hash table
			panic("too many probes")
		}
	}
}
