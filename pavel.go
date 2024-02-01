package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"math/bits"
	"os"
	"runtime"
	rtdebug "runtime/debug"
	"runtime/pprof"
	"runtime/trace"
	"sort"
	"sync"
	"time"
	"unsafe"
)

// ===== bits.go =====

// indexZero returns 0 if x does not have any zero bytes. Otherwise returns a
// non-zero integer, with pos*8 + 7 lowest bits unset, and (pos+1)*8 bit set.
func indexZero(x uint64) uint64 {
	// https://jameshfisher.com/2017/01/24/bitwise-check-for-zero-byte
	return (x - 0x0101010101010101) & ^x & 0x8080808080808080
}

func indexSemicolon(x uint64) uint64 {
	const allSemicolons = 0x3B3B3B3B3B3B3B3B
	return indexZero(x ^ allSemicolons)
}

func toNumber(word uint64) (int16, int) {
	// Possible numbers:
	//
	//	x.x
	//	xx.x
	//	-x.x
	//	-xx.x
	//
	// ASCII codes:
	//
	//	minus : 0x2D
	//	point : 0x2E
	//	digits: 0x30 ... 0x39
	//
	// Minus and point can be distinguished by 4th bit being 0, mask 0x10.
	// Byte 0 is either a digit, or minus. We can thus detect minus by checking
	// the 4th bit.
	//
	// Move 4th bit to 63, and back to 0.
	// Use signed int, so that right shift propagates the highest bit.
	not := ^word
	sign := (int64(not) << (63 - 4)) >> 63 // 0 or -1
	word ^= uint64(sign & ('-' ^ '0'))     // turn '-' into '0' if present
	// Now the word is in one of 3 forms: x.x, xx.x, xxx.x.
	//
	// The point can be between bytes bytes 1-3. It can also be distinguished
	// by mask 0x10. Its position will be the lowest non-zero byte, after we
	// mask out the digits. Byte 0 can be ignored.
	point := bits.TrailingZeros64(not & 0x10101000) // equals 12, 20, or 28
	// Shift the whole thing so that the point is now at byte 3.
	word <<= 28 ^ point // 28 ^ point == 28 - point

	// ASCII digits conveniently have high 4 bits being 0x3, so we can convert
	// them to decimals by &^ 0x30. This will have no effect on zero bytes.
	word &= ^uint64(0xFF<<24 | // remove point (byte 3)
		0xFFFFFF30FF3030FF) // convert ASCII digits to decimals (bytes 1, 2, 4)

	// The word now looks like (from low to high bytes): 00 AA BB 00 CC.
	// Our goal is to convert it to a decimal number: 100*AA + 10*BB + CC.
	//
	// Let's try to multiply the word by 100, 10, and 1, and see how we could
	// obtain the result:
	//
	//	                   |
	//	                   v
	//	    1x 00 AA BB 00 CC
	//	+  10x       00 AA BB 00 CC
	//	+ 100x          00 AA BB 00 CC
	//	                   |
	//	                   v
	//	           CC + 10*BB + 100*CC
	//
	// Consider the bytes of the sum. Bytes 0-3 are not greater than 90, so
	// will not "carry" to the right. Byte 4 potentially overflows, because its
	// max value is 999. At most 2 bits will be carried to byte 5. Byte 5
	// multiplies BB by 100 which is a multiple of 4, thus its value will have
	// 2 low bits "free", which conveniently can be occupied by the 2 bits
	// carried from byte 4. Bytes 6-7 are of no interest.
	//
	// This proves that byte 4 and low 2 bits of byte 5 exactly contain our
	// decimal number. To extract it, we can simply mask out bits [32, 42).
	const mix = 1 + 10<<16 + 100<<24
	number := ((word * mix) >> 32) & 0x3FF
	// It remains to take the sign into account. If sign == 0, we should return
	// the number as is. If sign == -1, we should return -number. We also know
	// that: -number == ^number + 1.
	//
	// We can use the sign to "conditionally" invert the number:
	//
	//	number ^ sign == number if sign == 0
	//	number ^ sign == ^number if sign == -1 (0xFFFF..FF)
	//
	// This is almost what we need, except we also need to add 1 for the
	// negative case (and only). Conveniently, -sign == (1 or 0), so adding
	// -sign does this "conditional" +1.
	return int16(int64(number) ^ sign - sign), point>>3 + 3
}

// ===== end of bits.go =====
// ===== buffers.go =====

type multiBuffer struct {
	bufs  [][]byte
	slots chan int
}

func newMultiBuffer(bufs [][]byte) *multiBuffer {
	slots := make(chan int, len(bufs))
	for i := range bufs {
		slots <- i
	}
	return &multiBuffer{bufs: bufs, slots: slots}
}

func (m *multiBuffer) get() buffer {
	id := <-m.slots
	return buffer{id: id, buf: m.bufs[id]}
}

func (m *multiBuffer) put(b buffer) {
	m.slots <- b.id
}

// ===== end of buffers.go =====
// ===== debug.go =====

const (
	debug         = false
	maxCollisions = 0
)

// ===== end of debug.go =====
// ===== hash.go =====

type hash = uint64

func intHash(x uint64) hash {
	// better but slower hash:
	//
	// const mult = 0x45D9F3B
	// x = ((x >> 17) ^ x) * mult
	// x = ((x >> 31) ^ x) * mult
	// return hash((x >> 13) ^ x)
	//
	// This one works good enough.
	x ^= x >> 13
	x ^= x >> 31
	return x
}

// [T any]
type T = stat

type entry struct {
	key  string64
	hash hash
	val  T
}

type stringMap struct {
	tab        []entry
	mask       uint64
	used       uint64
	collisions [maxCollisions]int
}

func newStringMap(size int) stringMap {
	// max load factor is a generous 1/16-1/8
	bits := bits.Len64(uint64(size-1)) + 3
	slots := 1 << bits
	return stringMap{mask: uint64(slots - 1), tab: make([]entry, slots)}
}

func (s *stringMap) grow() {
	if s.used++; s.used<<3 <= uint64(len(s.tab)) {
		return
	}
	s.mask = s.mask<<1 | 1
	tab := s.tab
	s.tab = make([]entry, s.mask+1)
	for _, e := range tab {
		if e.key.ptr != nil {
			*s.get(&e.key, e.hash) = e
		}
	}
}

func (s *stringMap) get(key *string64, h hash) *entry {
	for step, slot := 0, uint64(h)&s.mask; ; step++ {
		if sl := &s.tab[slot]; sl.key.equal(key) || sl.key.ptr == nil {
			if debug {
				s.collisions[step]++
			}
			return sl
		}
		slot = (slot + 1) & s.mask
	}
}

func (s *stringMap) getFast(h hash) *entry {
	return &s.tab[uint64(h)&s.mask]
}

// ===== end of hash.go =====
// ===== keys_pool.go =====

type keysPool struct {
	back [maxStations*maxNameLen/8 + 8]uint64
	next int
}

func (k *keysPool) get(words int) string64 {
	start := k.next
	k.next += words
	return string64{
		ptr:   (*uint64)(unsafe.Pointer(&k.back[start])),
		extra: max(words, 2) - 2,
	}
}

// ===== end of keys_pool.go =====
// ===== main.go =====

var (
	file       = flag.String("file", "measurements.txt", "the input file")
	workers    = flag.Int("w", 0, "the number of workers (NumCPU by default)")
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
	rttrace    = flag.String("trace", "", "write runtime trace to file")
	verbose    = flag.Bool("v", false, "pring debug logging")
)

const (
	maxStations = 10000
	maxNameLen  = 400 // 100 UTF-8 chars
	maxExtraLen = 7   // semicolon, minus, 3 digits, dot, newline

	numBuffers = 1024
	maxIOSize  = 512 << 10

	padding    = 16            // leave 2 uint64 words for sentinels
	maxLineLen = 512 - padding // must be >= maxNameLen + maxExtraLen

	// sysctl -a
	//	hw.cachelinesize: 128
	//	hw.l1icachesize: 131072
	//	hw.l1dcachesize: 65536
	//	hw.l2cachesize: 4194304
	//
	// bufferSize must be a multiple of 128
	bufferSize = maxLineLen + maxIOSize + padding
)

var back [numBuffers][bufferSize]byte
var output [maxLineLen * maxStations]byte

var logger *log.Logger

func start() {
	flag.Parse()
	writer := io.Discard
	if *verbose {
		writer = os.Stderr
	}
	logger = log.New(writer, ": ", log.Ltime|log.LUTC|log.Lmicroseconds|log.Lmsgprefix)

	rtdebug.SetGCPercent(-1) // disable GC
}

func main() {
	start()
	if path := *cpuprofile; path != "" {
		f, err := os.Create(path)
		noError(err)
		defer func() { noError(f.Close()) }()
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if path := *rttrace; path != "" {
		f, err := os.Create(path)
		noError(err)
		defer func() { noError(f.Close()) }()
		noError(trace.Start(f))
		defer trace.Stop()
	}

	cpus := runtime.NumCPU()
	runtime.GOMAXPROCS(cpus + 4)
	workers := *workers
	if workers == 0 {
		workers = cpus
	}
	logger.Println("workers", workers)

	start := time.Now()
	defer func() { logger.Println("running time", time.Now().Sub(start)) }()

	var bufs [numBuffers][]byte
	for i := range bufs {
		bufs[i] = back[i][:]
	}
	m := newMultiBuffer(bufs[:])

	jobs := make(chan job, numBuffers+256)

	var wg sync.WaitGroup
	ws := make([]*worker, workers)
	for i := range ws {
		w := newWorker(jobs, m)
		wg.Add(1)
		go func() {
			defer wg.Done()
			w.run()
		}()
		ws[i] = w
	}

	// runtime.LockOSThread()
	// defer runtime.UnlockOSThread()

	f1, err := os.Open(*file)
	noError(err)
	defer func() { noError(f1.Close()) }()
	stat, err := f1.Stat()
	noError(err)
	size := stat.Size()
	f2, err := os.Open(*file)
	noError(err)
	defer func() { noError(f2.Close()) }()

	mid := size / 2
	_, err = f2.Seek(mid, 0)
	noError(err)

	s1 := newScanner(f1, int(mid), maxIOSize)
	s2 := newScanner(f2, int(size-mid), maxIOSize)
	s1.load(0, maxLineLen)
	s2.load(maxLineLen, maxLineLen)

	var wg1 sync.WaitGroup
	wg1.Add(1)
	go func() {
		defer wg1.Done()
		lead(s2, m, jobs)
	}()
	lead(s1, m, jobs)
	wg1.Wait()
	close(jobs)
	wg.Wait()

	if *verbose {
		readT := s1.readT + s2.readT
		logger.Println("IO time", readT)
		logger.Printf("IO throughput %.3f GiB/s", float64(size)/float64(1<<30)/readT.Seconds())

		if debug {
			col := ws[0].stats.collisions
			for _, w := range ws[1:] {
				for i, c := range w.stats.collisions {
					col[i] += c
				}
			}
			logger.Printf("collisions: %+v\n", col)
		}
	}

	stats := ws[0].stats
	for _, w := range ws[1:] {
		stats.merge(w.stats)
	}
	fmt.Print(string(stats.print(output[:0])))
}

func noError(err error) {
	if err != nil {
		panic(err)
	}
}

// ===== end of main.go =====
// ===== scanner.go =====

type buffer struct {
	id  int
	buf []byte
}

type scanner struct {
	reader  io.Reader
	left    int
	maxRead int

	context []byte // last read bytes

	readT time.Duration
}

func newScanner(r io.Reader, total, maxRead int) *scanner {
	return &scanner{reader: r, left: total, maxRead: maxRead}
}

func (s *scanner) load(init, context int) {
	data := make([]byte, init, context)
	pos := 0
	for pos < init {
		n, err := s.reader.Read(data[pos:])
		if n == 0 || err == io.EOF {
			break
		}
		noError(err)
		pos += n
	}
	s.context = data[:pos]
}

func (s *scanner) read(to []byte) ([]byte, int) {
	if cap(to) < cap(s.context) {
		panic("input buffer too small")
	}
	to = to[:cap(to)]
	context := copy(to, s.context)
	pos := context
	for end := min(len(to), pos+min(s.maxRead, s.left)); pos < end; {
		// st := time.Now()
		n, err := s.reader.Read(to[pos:end])
		// s.readT += time.Now().Sub(st)
		if n == 0 || err == io.EOF {
			break
		}
		noError(err)
		s.left -= n
		pos += n
	}
	to = to[:pos]

	newContext := min(pos, cap(s.context))
	s.context = s.context[:newContext]
	copy(s.context, to[pos-newContext:pos])

	return to, context
}

// ===== end of scanner.go =====
// ===== stats.go =====

const initialSize = 4 << 10

type fastStats struct {
	stringMap //[stat]
}

func newFastStats() fastStats {
	return fastStats{stringMap: newStringMap(initialSize)}
}

func (s *fastStats) updateFast(name *string64, t int16, h hash, kp *keysPool) bool {
	if sl := s.getFast(h); sl.key.equalFast(name) {
		sl.val.update(t)
		return true
	}
	return false
}

func (s *fastStats) update(name *string64, t int16, h hash, kp *keysPool) {
	if sl := s.get(name, h); sl.key.ptr != nil {
		sl.val.update(t)
	} else {
		sl.key = kp.get(name.words())
		name.copy(&sl.key)
		sl.hash = h
		sl.val = newStat(t)
		s.grow()
	}
}

func (s *fastStats) merge(other fastStats) {
	for i := range other.tab {
		if e := &other.tab[i]; e.key.ptr != nil {
			if slot := s.get(&e.key, e.hash); slot.key.ptr != nil {
				slot.val.merge(e.val)
			} else {
				*slot = *e
				s.grow()
			}
		}
	}
}

func (s *fastStats) print(to []byte) []byte {
	type entry struct {
		slot int
		key  []byte
	}
	var back [maxStations]entry
	count := 0

	for i := range s.tab {
		if key := &s.tab[i].key; key.ptr != nil {
			back[count] = entry{slot: i, key: key.bytes()}
			count++
		}
	}
	sort.Slice(back[:count], func(i, j int) bool {
		return bytes.Compare(back[i].key, back[j].key) == -1
	})

	for _, e := range back[:count] {
		st := s.tab[e.slot].val
		key := e.key[:len(e.key)-1] // remove the semicolon
		to = append(append(to, key...), fmt.Sprintf("=%.1f/%.1f/%.1f\n",
			float64(st.minT)/10,
			float64(st.sumT)/float64(st.count)/10,
			float64(st.maxT)/10,
		)...)
	}
	return to
}

func printTemperature(t float64, to []byte) []byte {
	return append(to, fmt.Sprintf("%.1f", t/10)...)
}

type stat struct {
	count int32
	sumT  int64
	minT  int16
	maxT  int16
}

func newStat(t int16) stat {
	return stat{count: 1, sumT: int64(t), minT: t, maxT: t}
}

func (s *stat) update(t int16) {
	s.count++
	s.sumT += int64(t)
	if t < s.minT {
		s.minT = t
	} else if t > s.maxT {
		s.maxT = t
	}
}

func (s *stat) merge(other stat) {
	s.count += other.count
	s.sumT += other.sumT
	if o := other.minT; o < s.minT {
		s.minT = o
	}
	if o := other.maxT; o > s.maxT {
		s.maxT = o
	}
}

// ===== end of stats.go =====
// ===== string64.go =====

type string64 struct {
	first uint64  // the first word
	last  uint64  // the last word, or zero
	extra int     // number of words in between first and last
	ptr   *uint64 // all words
}

func (s string64) bytes() []byte {
	bytes := s.words() << 3
	last := s.last
	if last == 0 { // TODO: branchless
		if s.extra == 0 {
			last = s.first
		} else {
			last = *(*uint64)(unsafe.Add(unsafe.Pointer(s.ptr), s.extra<<3))
		}
	}
	bytes -= bits.LeadingZeros64(last) >> 3
	return unsafe.Slice((*byte)(unsafe.Pointer(s.ptr)), bytes)
}

func (s string64) words() int {
	return s.extra + 2 + int(s.first^(s.first-1))>>63 + int(s.last^(s.last-1))>>63
}

func (s string64) hash() hash {
	h := intHash(s.first)
	p := s.ptr
	for i := s.extra; i != 0; i-- {
		p = (*uint64)(unsafe.Add(unsafe.Pointer(p), 8))
		h ^= intHash(*p)
	}
	if s.last != 0 {
		h ^= intHash(s.last)
	}
	return h
}

func (s *string64) equalFast(other *string64) bool {
	return s.extra == 0 && other.extra == 0 && s.first == other.first && s.last == other.last
}

func (s *string64) equal(other *string64) bool {
	if s.first != other.first || s.last != other.last || s.extra != other.extra {
		return false
	}

	p1, p2 := s.ptr, other.ptr
	for i := s.extra; i != 0; i-- {
		p1 = (*uint64)(unsafe.Add(unsafe.Pointer(p1), 8))
		p2 = (*uint64)(unsafe.Add(unsafe.Pointer(p2), 8))
		if *p1 != *p2 {
			return false
		}
	}
	return true
}

// NB: to.ptr has >= s.words() words preallocated.
func (s *string64) copy(to *string64) {
	words := s.words()
	copy(unsafe.Slice(to.ptr, words), unsafe.Slice(s.ptr, words))
	to.first = s.first
	to.last = s.last
}

// ===== end of string64.go =====
// ===== worker.go =====

type job struct {
	buf     buffer
	context int
}

type worker struct {
	jobs  <-chan job
	bufs  *multiBuffer
	stats fastStats
	kp    keysPool
}

func newWorker(jobs <-chan job, bufs *multiBuffer) *worker {
	return &worker{jobs: jobs, bufs: bufs, stats: newFastStats()}
}

func (w *worker) run() {
	for job := range w.jobs {
		w.process(job.buf, job.context)
		w.bufs.put(job.buf)
	}
}

func lead(s *scanner, bufs *multiBuffer, jobs chan<- job) {
	for b := bufs.get(); ; b = bufs.get() {
		buf, context := s.read(b.buf)
		b.buf = buf
		jobs <- job{buf: b, context: context}
		if len(b.buf) == context { // nothing new
			break
		}
	}
}

func (w *worker) process(b buffer, context int) {
	buf := b.buf[lastNewLine(b.buf[:context])+1:] // no-op if not found
	buf = buf[:lastNewLine(buf)+1]                // empty if not found
	if len(buf) == 0 {
		return
	}
	if buf[len(buf)-1] != '\n' {
		buf = append(buf, 0) // A sentinel zero byte, in case ptr jumps past the end.
	}

	const parts = 4
	var bufs [4][]byte
	part := len(buf) / parts
	for i := 0; i < parts; i++ {
		nl := lastNewLine(buf[:part]) + 1
		bufs[i], buf = buf[:nl], buf[nl:]
	}
	buf1, buf2, buf3, buf4 := bufs[0], bufs[1], bufs[2], bufs[3]

	ptr1 := (*uint64)(unsafe.Pointer(&buf1[0]))
	ptr2 := (*uint64)(unsafe.Pointer(&buf2[0]))
	ptr3 := (*uint64)(unsafe.Pointer(&buf3[0]))
	ptr4 := (*uint64)(unsafe.Pointer(&buf4[0]))
	end1 := (*uint64)(unsafe.Add(unsafe.Pointer(ptr1), len(buf1)))
	end2 := (*uint64)(unsafe.Add(unsafe.Pointer(ptr2), len(buf2)))
	end3 := (*uint64)(unsafe.Add(unsafe.Pointer(ptr3), len(buf3)))
	end4 := (*uint64)(unsafe.Add(unsafe.Pointer(ptr4), len(buf4)))

	for ptr1 != end1 && ptr2 != end2 && ptr3 != end3 && ptr4 != end4 {
		name1 := string64{ptr: ptr1, first: *ptr1}
		name2 := string64{ptr: ptr2, first: *ptr2}
		name3 := string64{ptr: ptr3, first: *ptr3}
		name4 := string64{ptr: ptr4, first: *ptr4}

		idx1 := indexSemicolon(name1.first)
		idx2 := indexSemicolon(name2.first)
		idx3 := indexSemicolon(name3.first)
		idx4 := indexSemicolon(name4.first)

		var next1, next2, next3, next4 *uint64
		var h1, h2, h3, h4 hash

		if idx1 != 0 { // semicolon
			next1, h1 = parseOne(ptr1, idx1, &name1)
		} else {
			second := *(*uint64)(unsafe.Add(unsafe.Pointer(ptr1), 8))
			if idx := indexSemicolon(second); idx != 0 {
				next1, h1 = parseTwo(ptr1, second, idx, &name1)
			} else {
				next1, h1 = parseLong(ptr1, second, &name1)
			}
		}
		if idx2 != 0 { // semicolon
			next2, h2 = parseOne(ptr2, idx2, &name2)
		} else {
			second := *(*uint64)(unsafe.Add(unsafe.Pointer(ptr2), 8))
			if idx := indexSemicolon(second); idx != 0 {
				next2, h2 = parseTwo(ptr2, second, idx, &name2)
			} else {
				next2, h2 = parseLong(ptr2, second, &name2)
			}
		}
		if idx3 != 0 { // semicolon
			next3, h3 = parseOne(ptr3, idx3, &name3)
		} else {
			second := *(*uint64)(unsafe.Add(unsafe.Pointer(ptr3), 8))
			if idx := indexSemicolon(second); idx != 0 {
				next3, h3 = parseTwo(ptr3, second, idx, &name3)
			} else {
				next3, h3 = parseLong(ptr3, second, &name3)
			}
		}
		if idx4 != 0 { // semicolon
			next4, h4 = parseOne(ptr4, idx4, &name4)
		} else {
			second := *(*uint64)(unsafe.Add(unsafe.Pointer(ptr4), 8))
			if idx := indexSemicolon(second); idx != 0 {
				next4, h4 = parseTwo(ptr4, second, idx, &name4)
			} else {
				next4, h4 = parseLong(ptr4, second, &name4)
			}
		}

		value1, shift1 := toNumber(*next1)
		value2, shift2 := toNumber(*next2)
		value3, shift3 := toNumber(*next3)
		value4, shift4 := toNumber(*next4)

		if !w.stats.updateFast(&name1, value1, h1, &w.kp) {
			w.stats.update(&name1, value1, h1, &w.kp)
		}
		if !w.stats.updateFast(&name2, value2, h2, &w.kp) {
			w.stats.update(&name2, value2, h2, &w.kp)
		}
		if !w.stats.updateFast(&name3, value3, h3, &w.kp) {
			w.stats.update(&name3, value3, h3, &w.kp)
		}
		if !w.stats.updateFast(&name4, value4, h4, &w.kp) {
			w.stats.update(&name4, value4, h4, &w.kp)
		}

		ptr1 = (*uint64)(unsafe.Add(unsafe.Pointer(next1), shift1))
		ptr2 = (*uint64)(unsafe.Add(unsafe.Pointer(next2), shift2))
		ptr3 = (*uint64)(unsafe.Add(unsafe.Pointer(next3), shift3))
		ptr4 = (*uint64)(unsafe.Add(unsafe.Pointer(next4), shift4))
	}

	for ptr1 != end1 {
		name1 := string64{ptr: ptr1, first: *ptr1}
		idx1 := indexSemicolon(name1.first)
		var next1 *uint64
		var h1 hash
		if idx1 != 0 { // semicolon
			next1, h1 = parseOne(ptr1, idx1, &name1)
		} else {
			second := *(*uint64)(unsafe.Add(unsafe.Pointer(ptr1), 8))
			if idx := indexSemicolon(second); idx != 0 {
				next1, h1 = parseTwo(ptr1, second, idx, &name1)
			} else {
				next1, h1 = parseLong(ptr1, second, &name1)
			}
		}
		value1, shift1 := toNumber(*next1)
		if !w.stats.updateFast(&name1, value1, h1, &w.kp) {
			w.stats.update(&name1, value1, h1, &w.kp)
		}
		ptr1 = (*uint64)(unsafe.Add(unsafe.Pointer(next1), shift1))
	}

	for ptr2 != end2 {
		name2 := string64{ptr: ptr2, first: *ptr2}
		idx2 := indexSemicolon(name2.first)
		var next2 *uint64
		var h2 hash
		if idx2 != 0 { // semicolon
			next2, h2 = parseOne(ptr2, idx2, &name2)
		} else {
			second := *(*uint64)(unsafe.Add(unsafe.Pointer(ptr2), 8))
			if idx := indexSemicolon(second); idx != 0 {
				next2, h2 = parseTwo(ptr2, second, idx, &name2)
			} else {
				next2, h2 = parseLong(ptr2, second, &name2)
			}
		}
		value2, shift2 := toNumber(*next2)
		if !w.stats.updateFast(&name2, value2, h2, &w.kp) {
			w.stats.update(&name2, value2, h2, &w.kp)
		}
		ptr2 = (*uint64)(unsafe.Add(unsafe.Pointer(next2), shift2))
	}

	for ptr3 != end3 {
		name3 := string64{ptr: ptr3, first: *ptr3}
		idx3 := indexSemicolon(name3.first)
		var next3 *uint64
		var h3 hash
		if idx3 != 0 { // semicolon
			next3, h3 = parseOne(ptr3, idx3, &name3)
		} else {
			second := *(*uint64)(unsafe.Add(unsafe.Pointer(ptr3), 8))
			if idx := indexSemicolon(second); idx != 0 {
				next3, h3 = parseTwo(ptr3, second, idx, &name3)
			} else {
				next3, h3 = parseLong(ptr3, second, &name3)
			}
		}
		value3, shift3 := toNumber(*next3)
		if !w.stats.updateFast(&name3, value3, h3, &w.kp) {
			w.stats.update(&name3, value3, h3, &w.kp)
		}
		ptr3 = (*uint64)(unsafe.Add(unsafe.Pointer(next3), shift3))
	}

	for ptr4 != end4 {
		name4 := string64{ptr: ptr4, first: *ptr4}
		idx4 := indexSemicolon(name4.first)
		var next4 *uint64
		var h4 hash
		if idx4 != 0 { // semicolon
			next4, h4 = parseOne(ptr4, idx4, &name4)
		} else {
			second := *(*uint64)(unsafe.Add(unsafe.Pointer(ptr4), 8))
			if idx := indexSemicolon(second); idx != 0 {
				next4, h4 = parseTwo(ptr4, second, idx, &name4)
			} else {
				next4, h4 = parseLong(ptr4, second, &name4)
			}
		}
		value4, shift4 := toNumber(*next4)
		if !w.stats.updateFast(&name4, value4, h4, &w.kp) {
			w.stats.update(&name4, value4, h4, &w.kp)
		}
		ptr4 = (*uint64)(unsafe.Add(unsafe.Pointer(next4), shift4))
	}
}

func parseOne(ptr *uint64, idx uint64, name *string64) (*uint64, hash) {
	pos := bits.TrailingZeros64(idx)
	name.first &= 1<<pos - 1
	return (*uint64)(unsafe.Add(unsafe.Pointer(ptr), (pos>>3)+1)), intHash(name.first)
}

func parseTwo(ptr *uint64, second uint64, idx uint64, name *string64) (*uint64, hash) {
	pos := bits.TrailingZeros64(idx)
	second &= 1<<pos - 1
	name.last = second
	return (*uint64)(unsafe.Add(unsafe.Pointer(ptr), (pos>>3)+9)), intHash(name.first) ^ intHash(second)
}

func parseLong(ptr *uint64, second uint64, str *string64) (next *uint64, h hash) {
	h = intHash(str.first) ^ intHash(second)
	ptr = (*uint64)(unsafe.Add(unsafe.Pointer(ptr), 16))
	for word := *ptr; ; word = *ptr {
		str.extra++
		idx := indexSemicolon(word)
		if idx == 0 { // no semicolon
			h ^= intHash(word)
			ptr = (*uint64)(unsafe.Add(unsafe.Pointer(ptr), 8))
			continue
		}
		pos := bits.TrailingZeros64(idx)
		word &= uint64(1)<<pos - 1
		str.last = word
		h ^= intHash(word)
		ptr = (*uint64)(unsafe.Add(unsafe.Pointer(ptr), (pos>>3)+1))
		return ptr, h
	}
}

func lastNewLine(buf []byte) int {
	for p := len(buf) - 1; p >= 0; p-- {
		if buf[p] == '\n' {
			return p
		}
	}
	return -1
}

// ===== end of worker.go =====
