package main

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"syscall"
)

// how fast can we do just the io part?

const (
	chunk         = 1 << 24
	workersPerCPU = 1
)

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

	workers := runtime.NumCPU() * workersPerCPU

	// build a queue of file chunks
	chunks := make(chan []byte, len(data)/chunk+1+workers)
	for i := 0; i < len(data); i += chunk {
		next := i + chunk
		if next > len(data) {
			next = len(data)
		}
		chunks <- data[i:next]
	}
	for i := 0; i < workers; i++ {
		chunks <- nil
	}

	// run workers
	var total uint64
	var wait sync.WaitGroup
	for i := 0; i < workers; i++ {
		wait.Add(1)
		go parse(chunks, &total, &wait)
	}

	wait.Wait()
	fmt.Println(total)
}

func parse(chunks <-chan []byte, total *uint64, wait *sync.WaitGroup) {
	defer wait.Done()
	var subtotal int
	for subdata := range chunks {
		if subdata == nil {
			// no more chunks
			atomic.AddUint64(total, uint64(subtotal))
			return
		}
		subtotal += bytes.Count(subdata, []byte{'\n'})
	}
}
