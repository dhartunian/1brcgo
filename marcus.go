package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"slices"
	"sync"
	"unsafe"
)

const (
	inputFile = "measurements.txt"
	// Set profile to true to generate CPU and heap profiles.
	profile = false
	// Set runTests to run tests at the start of the program.
	runTests = false
	// numGoroutines controls the number of goroutines to spawn to scan the
	// input file.
	// The processor has 8 cores + 2 efficiency cores.
	numGoroutines = 10
)

func main() {
	// Disable GC. We don't need that!
	debug.SetGCPercent(-1)

	if runTests {
		// Run tests.
		TestSplitCityAndTemp()
	}

	if profile {
		// Profile CPU.
		pf, err := os.Create("cpu.pb")
		if err != nil {
			panic(err)
		}
		defer pf.Close()
		if err := pprof.StartCPUProfile(pf); err != nil {
			panic(err)
		}
		defer pprof.StopCPUProfile()
	}

	// Run the program.
	Run()

	if profile {
		// Profile memory.
		mf, err := os.Create("mem.pb")
		if err != nil {
			panic(err)
		}
		defer mf.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(mf); err != nil {
			panic(err)
		}
	}
}

// Run runs the program.
func Run() {
	// Determine the size of the file.
	info, err := os.Stat(inputFile)
	if err != nil {
		panic(err)
	}
	size := info.Size()
	chunkSize := size / numGoroutines

	var m0, m1, m2, m3, m4, m5, m6, m7, m8, m9 *SummaryMap
	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		start := int64(i) * chunkSize
		end := int64(i+1) * chunkSize

		wg.Add(1)
		go func(i int, start, end int64) {
			// Allocate a summary map.
			var m SummaryMap
			switch i {
			case 0:
				m0 = &m
			case 1:
				m1 = &m
			case 2:
				m2 = &m
			case 3:
				m3 = &m
			case 4:
				m4 = &m
			case 5:
				m5 = &m
			case 6:
				m6 = &m
			case 7:
				m7 = &m
			case 8:
				m8 = &m
			case 9:
				m9 = &m
			}

			// Open a new handle to the file. We don't bother closing it.
			reader, err := os.Open(inputFile)
			if err != nil {
				panic(err)
			}
			if _, err = reader.Seek(start, 0); err != nil {
				panic(err)
			}

			// Scan the records.
			var s Scanner
			s.Init(reader, make([]byte, 5*1024*1024 /* 5MB */))
			if start != 0 {
				// Skip the first line unless this is the goroutine reading from
				// the beginning. The previous goroutine will scan the skipped
				// line.
				_, _ = s.Scan()
			}
			bytesToRead := int(end - start)
			for line, ok := s.Scan(); ok; line, ok = s.Scan() {
				city, temp := SplitCityAndTemp(line)
				rec := m.Lookup(city)
				rec.Add(temp)
				// Scan just past the number of bytes to read because the
				// goroutine reading from the next section will skip the first
				// line it reads.
				if s.TotalBytesScanned() > bytesToRead {
					break
				}
			}
			wg.Done()
		}(i, start, end)
	}

	// Wait for the scanners to finish.
	wg.Wait()

	// Put the maps in an array so that they are easier to work with.
	maps := [...]*SummaryMap{m0, m1, m2, m3, m4, m5, m6, m7, m8, m9}

	// Count the number of total cities collected.
	numAllCities := 0
	for i := range maps {
		numAllCities += len(maps[i].Keys())
	}

	// Allocate a slice to hold all possible cities.
	cities := make([]string, numAllCities)

	// Copy the cities from each map into the new slice.
	tmp := cities
	for i := range maps {
		c := maps[i].Keys()
		copy(tmp, c)
		tmp = tmp[len(c):]
	}

	// Sort and de-duplicate the cities.
	slices.Sort(cities)
	cities = slices.Compact(cities)

	// Print the output.
	w := bufio.NewWriter(os.Stdout)
	for _, city := range cities {
		var s Summary
		// Merge all summaries for this city together.
		for i := range maps {
			if sub := maps[i].LookupExisting(city); sub != nil {
				s.Merge(sub)
			}
		}
		_, err := fmt.Fprintf(w, "%s=%.1f/%.1f/%.1f\n", city, s.Min(), s.Avg(), s.Max())
		if err != nil {
			panic(err)
		}
	}
	if err := w.Flush(); err != nil {
		panic(err)
	}
}

// SplitCityAndTemp splits the given line into a city and temperature and
// returns both.
func SplitCityAndTemp(line []byte) (city []byte, _ Temp) {
	// Gather the key and fetch the corresponding record. We can do this without
	// scanning the line because there are only four possible positions for
	// the semicolon. The valid formats for the line are:
	//
	//     x;0.0
	//    x;00.0
	//    x;-0.0
	//   x;-00.0
	//
	// We use the limited locations of semicolons and minus characters to avoid
	// conditional expressions and loops.
	l := len(line)
	// In the common case, the line will be more than 5 characters.
	if l > 5 {
		// Avoid future bounds checks.
		_ = line[:l-6]

		// Gather all the digits. The tenths digit is the last one, the decimal
		// point is skipped, the ones digit is the third to last character, and
		// the tens digit is the fourth to last character, if it exists. Further
		// down we'll zero tens if it was actually a minus or semicolon.
		tenths := int(line[l-1] - '0')
		ones := int(line[l-3]-'0') * 10
		tens := int(line[l-4]-'0') * 100

		// neg4 is 1 if a minus character is present at len-4, and 0 otherwise.
		// neg5 is 1 if a minus character is present at len-5, and 0 otherwise.
		//
		// NOTE: The Go compiler eliminates jumps when using this form of
		// conditional.
		var neg4 int
		var neg5 int
		if line[l-4] == '-' {
			neg4 = 1
		}
		if line[l-5] == '-' {
			neg5 = 1
		}
		// neg is 1 if a minus character is present at len-4 or len-5.
		neg := neg4 | neg5

		// semi4 is 1 if a semi-column is present at len-4, and 0 otherwise.
		//
		// NOTE: The Go compiler eliminates jumps when using this form.
		var semi4 int
		if line[l-4] == ';' {
			semi4 = 1
		}

		// Clear tens if there was a minus character in that position.
		tens = tens &^ -neg4

		// Clear tens if there was a semicolon in that position.
		tens = tens &^ -semi4

		// Add the ones and tenths digits.
		t := ones + tenths

		// Add the tens digit.
		t += tens

		// Negate the value if we found a minus character in len-4 or len-5
		// position.
		//
		// See "conditionally negate a value without branching", from "Bit
		// Twiddling Hacks",
		// https://graphics.stanford.edu/~seander/bithacks.html.
		t = (t ^ -neg) + neg

		// TODO: There's another bounds check here even in cases where it's
		// fairly obvious to me that semiPos is 4, 5, or 6 (the current
		// commented-out semiPos doesn't make that so obvious, but other ways of
		// calculating it do). But I can't convince the Go compiler of this, and
		// the switch with some jumps is faster.
		//
		// semi5 is 1 if a semi-column is present at len-4, and 0 otherwise.
		// var semi5 int
		// if line[l-5] == ';' {
		// 	semi5 = 1
		// }
		// semiPos1 := 4
		// semiPos2 := semi5 &^ semi4
		// semiPos3 := 2 & ^((semi5 | semi4) << 1)
		// semiPos := l - (semiPos1 + semiPos2 + semiPos3)
		// return line[:semiPos], Temp(t)

		switch {
		case line[l-4] == ';':
			return line[:l-4], Temp(t)
		case line[l-5] == ';':
			return line[:l-5], Temp(t)
		default:
			return line[:l-6], Temp(t)
		}
	} else {
		// Special case. We only have 5 bytes, which must be in the form
		// "x;0.0". So the key is the first byte and the semicolon is the
		// second.

		// Avoid future bounds checks.
		_ = line[:l-5]

		// Gather the two digits.
		tenths := int(line[l-1] - '0')
		ones := int(line[l-3]-'0') * 10
		t := ones + tenths

		return line[:l-4], Temp(t)
	}
}

const (
	// Multi-byte characters are rare, so we'll group them in with other
	// characters by ignoring the most significant bit. Our table can then be
	// 128x128x128.
	tableDimension = 128
)

// SummaryMap is a map from a byte slice key to a Summary.
type SummaryMap struct {
	// This is a ~120MB table.
	table [tableDimension][tableDimension][tableDimension]SummaryList
	keys  []string
}

// Keys returns all the keys in the map, as strings.
func (rt *SummaryMap) Keys() []string {
	return rt.keys
}

// LookupExisting returns the Summary in the map associated with the key. If
// there is no summary matching the key, it returns nil.
func (rt *SummaryMap) LookupExisting(s string) *Summary {
	p := unsafe.StringData(s)
	key := unsafe.Slice(p, len(s))

	b0 := (key[0]) & 127
	// Use the NULL byte for the second and third characters in the lookup table
	// if the key is less than 3 bytes long.
	b1 := byte(0)
	b2 := byte(0)
	if len(key) > 2 {
		b2 = (key[2]) & 127
	}
	if len(key) > 1 {
		b1 = (key[1]) & 127
	}

	// Use the list with the matching first three bytes.
	list := &rt.table[b0][b1][b2]

	// Check the first summary in the list.
	rec := &list.first
	// The Go compiler does not allocate in this special case. See
	// https://github.com/golang/go/blob/e9b3ff15/src/bytes/bytes.go#L19
	if string(key) == rec.key {
		return rec
	}

	// Check the other record in the list.
	for i := range list.others {
		rec := &list.others[i]
		if string(key) == rec.key {
			return rec
		}
	}

	// The summary was not found.
	return nil
}

// Lookup returns the Summary in the map associated with key. If one does not
// exist, a new one is added to the map and returned.
//
// The given key must be non-empty.
func (rt *SummaryMap) Lookup(key []byte) *Summary {
	b0 := (key[0]) & 127
	// Use the NULL byte for the second and third characters in the lookup table
	// if the key is less than 3 bytes long.
	b1 := byte(0)
	b2 := byte(0)
	if len(key) > 2 {
		b2 = (key[2]) & 127
	}
	if len(key) > 1 {
		b1 = (key[1]) & 127
	}

	// Use the list with the matching first three bytes.
	list := &rt.table[b0][b1][b2]

	// Check the first summary in the list.
	rec := &list.first
	// The Go compiler does not allocate in this special case. See
	// https://github.com/golang/go/blob/e9b3ff15/src/bytes/bytes.go#L19
	if string(key) == rec.key {
		return rec
	}

	// Check the other record in the list.
	for i := range list.others {
		rec := &list.others[i]
		if string(key) == rec.key {
			return rec
		}
	}

	// The summary was not found. Allocate a string for the key, and add a new
	// summary to the map.
	city := string(key)
	rt.keys = append(rt.keys, city)
	return list.Append(city)
}

// SummaryList is a list of Summary structs.
type SummaryList struct {
	// TODO(mgartner): Not sure this fast-path really does anything. Test it.
	first  Summary
	others []Summary
}

// Append adds a new Summary to the list with the given key and returns the
// newly created Summary.
func (l *SummaryList) Append(key string) *Summary {
	r := Summary{
		key: key,
		min: math.MaxInt16,
		max: math.MinInt16,
	}
	if len(l.first.key) == 0 {
		l.first = r
		return &l.first
	}
	l.others = append(l.others, r)
	return &l.others[len(l.others)-1]
}

// Temp represents a temperature, stored as an integer 10x the temperature.
type Temp int16

// AsFloat returns the temperature as a float64.
func (t Temp) AsFloat() float64 {
	return float64(t) / 10.0
}

// TempSum represents a sum of temperatures, stored as an integer 10x the
// temperature.
type TempSum int64

// AsFloat returns the temperature as a float64.
func (t TempSum) AsFloat() float64 {
	return float64(t) / 10.0
}

// Summary summarizes all temperature recordings for a given key.
//
// NOTE: The order of fields is important for maintaining a struct of 32 bytes.
type Summary struct {
	key   string
	sum   TempSum
	count uint32
	min   Temp
	max   Temp
}

// Add adds a new temperature to the summary.
func (r *Summary) Add(t Temp) {
	r.min = min(r.min, t)
	r.max = max(r.max, t)
	r.sum += TempSum(t)
	r.count++
}

// Merge merges other with the record.
func (r *Summary) Merge(other *Summary) {
	r.min = min(r.min, other.min)
	r.max = max(r.max, other.max)
	r.sum += other.sum
	r.count += other.count
}

// Min returns the minimum temperature as a float64.
func (r Summary) Min() float64 {
	return r.min.AsFloat()
}

// Max returns the maximum temperature as a float64.
func (r Summary) Max() float64 {
	return r.max.AsFloat()
}

// Avg returns the mean temperature as float64.
func (r Summary) Avg() float64 {
	return r.sum.AsFloat() / float64(r.count)
}

// Scanner is copied from bufio.Scanner, with bunch of not-so-necessary
// funcationality ripped out.
type Scanner struct {
	r                 io.Reader // The reader provided by the client.
	buf               []byte    // Buffer used as argument to split.
	start             int       // First non-processed byte in buf.
	end               int       // End of data in buf.
	totalBytesScanned int
}

// Init initializes the scanner with the given buffer.
func (s *Scanner) Init(r io.Reader, buf []byte) {
	s.r = r
	s.buf = buf
}

// Scan scans the io.Reader for the next line and returns it.
func (s *Scanner) Scan() (line []byte, ok bool) {
	// Loop until we have a line.
	for {
		// See if we can get a line with what we already have.
		// If we've run out of data but have an error, give the split function
		// a chance to recover any remaining, possibly empty line.
		if s.end > s.start {
			var advance int
			advance, line = scanLines(s.buf[s.start:s.end])
			if !s.advance(advance) {
				return nil, false
			}
			if line != nil {
				s.totalBytesScanned += advance
				return line, true
			}
		}
		// Must read more data.
		// First, shift data to beginning of buffer if there's lots of empty space
		// or space is needed.
		if s.start > 0 && (s.end == len(s.buf) || s.start > len(s.buf)/2) {
			copy(s.buf, s.buf[s.start:s.end])
			s.end -= s.start
			s.start = 0
		}
		// Finally we can read some input.
		n, err := s.r.Read(s.buf[s.end:len(s.buf)])
		s.end += n
		if err != nil {
			return nil, false
		}
	}
}

// TotalBytesScanned returns the total number of bytes scanned from all Scan
// calls.
func (s *Scanner) TotalBytesScanned() int {
	return s.totalBytesScanned
}

func (s *Scanner) advance(n int) bool {
	if n < 0 {
		return false
	}
	if n > s.end-s.start {
		return false
	}
	s.start += n
	return true
}

func scanLines(data []byte) (advance int, token []byte) {
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		// We have a full newline-terminated line.
		return i + 1, data[0:i]
	}
	// Request more data.
	return 0, nil
}

// /////////////////////////////////////////////////////////
// Tests
// /////////////////////////////////////////////////////////

func TestSplitCityAndTemp() {
	type testCase struct {
		input        []byte
		expectedCity []byte
		expectedTemp Temp
	}
	testCases := []testCase{
		{[]byte("x;1.2"), []byte("x"), 12},
		{[]byte("x;-1.2"), []byte("x"), -12},
		{[]byte("x;12.3"), []byte("x"), 123},
		{[]byte("x;-12.3"), []byte("x"), -123},
		{[]byte("foo;99.9"), []byte("foo"), 999},
		{[]byte("foo;-99.9"), []byte("foo"), -999},
		{[]byte("abcdefghijklmnop;0.6"), []byte("abcdefghijklmnop"), 6},
		{[]byte("abcdefghijklmnop;-0.6"), []byte("abcdefghijklmnop"), -6},
		{[]byte("abcdefghijklmnop;12.6"), []byte("abcdefghijklmnop"), 126},
		{[]byte("abcdefghijklmnop;-12.6"), []byte("abcdefghijklmnop"), -126},
		{[]byte("é;8.9"), []byte("é"), 89},
		{[]byte("é;78.9"), []byte("é"), 789},
		{[]byte("é;-8.9"), []byte("é"), -89},
		{[]byte("é;-78.9"), []byte("é"), -789},
		{[]byte("üøª∂˙ßªß∂˙ªº∂ß∂˙ªº;5.6"), []byte("üøª∂˙ßªß∂˙ªº∂ß∂˙ªº"), 56},
		{[]byte("üøª∂˙ßªß∂˙ªº∂ß∂˙ªº;45.6"), []byte("üøª∂˙ßªß∂˙ªº∂ß∂˙ªº"), 456},
		{[]byte("üøª∂˙ßªß∂˙ªº∂ß∂˙ªº;-5.6"), []byte("üøª∂˙ßªß∂˙ªº∂ß∂˙ªº"), -56},
		{[]byte("üøª∂˙ßªß∂˙ªº∂ß∂˙ªº;-45.6"), []byte("üøª∂˙ßªß∂˙ªº∂ß∂˙ªº"), -456},
	}
	for _, tc := range testCases {
		city, temp := SplitCityAndTemp(tc.input)
		if string(city) != string(tc.expectedCity) {
			panic(fmt.Sprintf("expected key %q, got %q", tc.expectedCity, city))
		}
		if temp != tc.expectedTemp {
			panic(fmt.Sprintf("expected temp %d, got %d", tc.expectedTemp, temp))
		}
	}
}
