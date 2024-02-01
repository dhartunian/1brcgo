// 2024, michae2

package main

import (
	"bytes"
	"fmt"
	"hash/maphash"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"syscall"
)

const (
	tableN    = 1 << 11
	tableMask = 1<<11 - 1
	chunk     = 1 << 21
	maxProbe  = 1 << 5
)

type record struct {
	// maybe inline the key?
	key []byte
	hsh uint64
	sum int64
	cnt uint32
	min int16
	max int16
}

func (r *record) String() string {
	return fmt.Sprintf(
		"%s=%.1f/%.1f/%.1f\n",
		r.key, float64(r.min)/10.0, float64(r.sum)/float64(r.cnt)/10.0, float64(r.max)/10.0,
	)
}

type table []record

func main() {
	//syscall.Setpriority(syscall.PRIO_PROCESS, 0, -20)

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
		print(res)
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
				for j := uint64(0); j < maxProbe; j++ {
					// linear probing
					resultRec := &results[int((rec.hsh+j)&tableMask)]
					if resultRec.cnt == 0 {
						*resultRec = *rec
						continue next_bucket
					} else if resultRec.hsh != rec.hsh {
						//} else if !bytes.Equal(resultRec.key, rec.key) {
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

			key := off

			// find the separator
			// this is the most expensive part of the whole thing

			/*
				for ; off+8 <= len(subdata); off += 8 {
					_ = subdata[off+7]
					if subdata[off+0] == ';' {
						goto found
					}
					if subdata[off+1] == ';' {
						off += 1
						goto found
					}
					if subdata[off+2] == ';' {
						off += 2
						goto found
					}
					if subdata[off+3] == ';' {
						off += 3
						goto found
					}
					if subdata[off+4] == ';' {
						off += 4
						goto found
					}
					if subdata[off+5] == ';' {
						off += 5
						goto found
					}
					if subdata[off+6] == ';' {
						off += 6
						goto found
					}
					if subdata[off+7] == ';' {
						off += 7
						goto found
					}
				}
			*/

			/*
				for subdata[off] != ';' {
					off++
				}
			*/

			//found:

			/*
				for ; off < len(subdata); off++ {
					if subdata[off] == ';' {
						break
					}
				}
			*/

			/*
				for i := range subdata[off:] {
					if subdata[off+i] == ';' {
						off += i
						break
					}
				}
			*/

			off += bytes.IndexByte(subdata[off:], ';')
			sep := off
			// skip over the ;
			off++

			/*
					for ; off+8 < len(subdata); off += 8 {
						word := binary.LittleEndian.Uint64(subdata[off : off+8])
						if word&0xff == 0x3b {
							goto found
						}
						if word&0xff00 == 0x3b00 {
							off += 1
							goto found
						}
						if word&0xff0000 == 0x3b0000 {
							off += 2
							goto found
						}
						if word&0xff000000 == 0x3b000000 {
							off += 3
							goto found
						}
						if word&0xff00000000 == 0x3b00000000 {
							off += 4
							goto found
						}
						if word&0xff0000000000 == 0x3b0000000000 {
							off += 5
							goto found
						}
						if word&0xff000000000000 == 0x3b000000000000 {
							off += 6
							goto found
						}
						if word&0xff00000000000000 == 0x3b00000000000000 {
							off += 7
							goto found
						}
					}
					for subdata[off] != ';' {
						off++
					}
				found:
					sep := off
					// skip over the ;
					off++
			*/

			// hash the key
			hash := maphash.Bytes(seed, subdata[key:sep])

			/*
				hash := uint64(2003)
				k := key
				for ; k+8 <= sep; k += 8 {
					hash ^= binary.NativeEndian.Uint64(subdata[k : k+8])
					hash ^= hash >> 47
					hash *= 0xc6a4a7935bd1e995
				}
				switch sep - k {
				case 7:
					hash ^= uint64(subdata[k]) << 48
					k++
					fallthrough
				case 6:
					hash ^= uint64(subdata[k]) << 40
					k++
					fallthrough
				case 5:
					hash ^= uint64(subdata[k]) << 32
					k++
					fallthrough
				case 4:
					hash ^= uint64(binary.NativeEndian.Uint32(subdata[k : k+4]))
				case 3:
					hash ^= uint64(subdata[k]) << 16
					k++
					fallthrough
				case 2:
					hash ^= uint64(subdata[k]) << 8
					k++
					fallthrough
				case 1:
					hash ^= uint64(subdata[k])
				}
				hash *= 0xc6a4a7935bd1e995
			*/

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
					rec.key = subdata[key:sep]
					rec.hsh = hash
					rec.sum = int64(temp)
					rec.cnt = 1
					rec.min = int16(temp)
					rec.max = int16(temp)
					continue next_line
				} else if rec.hsh != hash {
					//} else if !bytes.Equal(rec.key, subdata[key:sep]) {
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
