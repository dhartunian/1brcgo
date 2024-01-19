package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"slices"
	"sync"
)

const (
	inputFile = "measurements.txt"
	// Set profile to true to generate CPU and heap profiles.
	profile = false
	// Set runTestsAndAssertions to run tests at the start of the program and
	// perform runtime assertions.
	runTestsAndAssertions = false
	// numGoroutines controls the number of goroutines to spawn to scan the
	// input file.
	// The processor has 8 cores + 2 efficiency cores.
	numGoroutines = 10
)

func main() {
	// Disable GC. We don't need that!
	debug.SetGCPercent(-1)

	if runTestsAndAssertions {
		// Run tests.
		TestParseTemp()
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
				s.SkipFirstLine()
			}
			bytesToRead := int(end - start)
			for city, temp, ok := s.Scan(); ok; city, temp, ok = s.Scan() {
				t := ParseTemp(temp)
				sum := m.Lookup(city)
				sum.Add(t)
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
		s := Summary{
			key: city,
			min: math.MaxInt16,
			max: math.MinInt16,
		}
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

// ParseTemp parses a byte slice representing a temperature.
func ParseTemp(t []byte) Temp {
	// Gather the key and fetch the corresponding record. We can do this without
	// scanning the line because there are only four possible positions for
	// the semicolon. The valid formats for the line are:
	//
	//     0.0
	//    00.0
	//    -0.0
	//   -00.0
	//
	// We use the limited locations of semicolons and minus characters to avoid
	// conditional expressions and loops.
	l := len(t)
	switch l {
	case 3:
		tenths := int(t[2] - '0')
		ones := int(t[0]-'0') * 10
		return Temp(ones + tenths)
	case 4:
		tenths := int(t[3] - '0')
		ones := int(t[1]-'0') * 10
		tens := int(t[0]-'0') * 100

		// neg is 1 if the first character is the minus charater, and 0
		// otherwise.
		//
		// NOTE: The Go compiler eliminates jumps when using this form of
		// conditional.
		var neg int
		if t[0] == '-' {
			neg = 1
		}

		// Clear tens if there was a minus character in that position.
		tens = tens &^ -neg

		// Add the ones and tenths digits.
		temp := ones + tenths

		// Add the tens digit.
		temp += tens

		// Negate the value if we found a minus character in len-4 or len-5
		// position.
		//
		// See "conditionally negate a value without branching", from "Bit
		// Twiddling Hacks",
		// https://graphics.stanford.edu/~seander/bithacks.html.
		temp = (temp ^ -neg) + neg

		return Temp(temp)
	case 5:
		tenths := int(t[4] - '0')
		ones := int(t[2]-'0') * 10
		tens := int(t[1]-'0') * 100
		temp := tens + ones + tenths
		temp *= -1
		return Temp(temp)
	default:
		// Impossible.
		if runTestsAndAssertions {
			panic(fmt.Sprintf("unexpected temperature slice length %d", l))
		}
		return Temp(0)
	}
}

const (
	// ASCII characters are all less than 128. Multi-byte characters, which
	// start with bytes greater than 128 are rare, so we'll group them in with
	// other characters by ignoring the most significant bit. Our table can then
	// be 128x128x128.
	tableDimension = 128
)

// SummaryMap is a map from a 3-byte key to a Summary linked list.
type SummaryMap struct {
	table [tableDimension][tableDimension][tableDimension]*Summary
	keys  []string
}

// Keys returns all the keys in the map, as strings.
func (m *SummaryMap) Keys() []string {
	return m.keys
}

// LookupExisting returns the Summary in the map associated with the key. If
// there is no summary matching the key, it returns nil.
func (m *SummaryMap) LookupExisting(key string) *Summary {
	var b0, b1, b2 byte
	b0 = key[0] & 127
	// Use the NULL byte for the second and third characters in the lookup table
	// if the key is less than 3 bytes long.
	if len(key) > 1 {
		b1 = key[1] & 127
	}
	if len(key) > 2 {
		b2 = key[2] & 127
	}

	s := m.table[b0][b1][b2]
	for s != nil {
		if key == s.key {
			return s
		}
		s = s.next
	}

	// The summary was not found.
	return nil
}

// Lookup returns the Summary in the map associated with key. If one does not
// exist, a new one is added to the map and returned.
//
// The given key must be non-empty.
func (m *SummaryMap) Lookup(key []byte) *Summary {
	var b0, b1, b2 byte
	b0 = key[0] & 127
	// Use the NULL byte for the second and third characters in the lookup table
	// if the key is less than 3 bytes long.
	if len(key) > 1 {
		b1 = key[1] & 127
	}
	if len(key) > 2 {
		b2 = key[2] & 127
	}

	s := m.table[b0][b1][b2]
	last := s
	for s != nil {
		// The Go compiler does not allocate in this special case. See
		// https://github.com/golang/go/blob/e9b3ff15/src/bytes/bytes.go#L19
		if string(key) == s.key {
			return s
		}
		last = s
		s = s.next
	}

	// The summary was not found. Allocate a string for the key, and add a new
	// summary to the map.
	s = &Summary{
		key: string(key),
		min: math.MaxInt16,
		max: math.MinInt16,
	}
	if last != nil {
		last.next = s
	} else {
		m.table[b0][b1][b2] = s
	}
	m.keys = append(m.keys, s.key)
	return s
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
type Summary struct {
	key   string
	next  *Summary
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

// Merge merges other with the Summary.
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
// functionality ripped out.
type Scanner struct {
	r                 io.Reader // The reader provided by the client.
	buf               []byte    // Buffer used as argument to split.
	start             int       // First non-processed byte in buf.
	end               int       // End of data in buf.
	totalBytesScanned int
	firstLineRead     bool
}

// Init initializes the scanner with the given buffer.
func (s *Scanner) Init(r io.Reader, buf []byte) {
	s.r = r
	s.buf = buf
}

// SkipFirstLine scans until the first newline in the io.Reader.
func (s *Scanner) SkipFirstLine() {
	if s.firstLineRead {
		panic("cannot call SkipFirstLine multiple times not after Scan")
	}
	s.firstLineRead = true
	// Perform the first read.
	n, err := s.r.Read(s.buf)
	s.end += n
	if err != nil {
		panic(err)
	}
	advance := scanLine(s.buf)
	if !s.advance(advance) {
		panic("could not advance")
	}
}

// Scan scans the io.Reader for the next line and returns it.
func (s *Scanner) Scan() (city []byte, temp []byte, ok bool) {
	s.firstLineRead = true
	// Loop until we have a full line.
	for {
		// See if we can get a line with what we already have.
		if s.end > s.start {
			var advance int
			advance, city, temp = scanCityAndTemp(s.buf[s.start:s.end])
			if !s.advance(advance) {
				return nil, nil, false
			}
			if city != nil {
				s.totalBytesScanned += advance
				return city, temp, true
			}
		}
		// Must read more data. First, shift data to the beginning of the
		// buffer.
		if s.start > 0 {
			copy(s.buf, s.buf[s.start:s.end])
			s.end -= s.start
			s.start = 0
		}
		// Read some input.
		n, err := s.r.Read(s.buf[s.end:len(s.buf)])
		s.end += n
		if err != nil {
			return nil, nil, false
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

func scanLine(data []byte) (advance int) {
	for i := range data {
		if data[i] == '\n' {
			return i + 1
		}
	}
	return 0
}

func scanCityAndTemp(data []byte) (advance int, city []byte, temp []byte) {
	i := 0
	for ; i < len(data); i++ {
		if data[i] == ';' {
			city = data[:i]
			break
		}
	}
	cityEnd := i
	// The temperature is at least 3 bytes, e.g., "0.0", so skip ahead 4 bytes
	// to start looking for the newline.
	i += 4
	for ; i < len(data); i++ {
		if data[i] == '\n' {
			temp = data[cityEnd+1 : i]
			return i + 1, city, temp
		}
	}
	return 0, nil, nil
}

// /////////////////////////////////////////////////////////
// Tests
// /////////////////////////////////////////////////////////

func TestParseTemp() {
	type testCase struct {
		input        []byte
		expectedTemp float64
	}
	testCases := []testCase{
		{[]byte("0.0"), 0},
		{[]byte("0.1"), 0.1},
		{[]byte("0.5"), 0.5},
		{[]byte("0.9"), 0.9},
		{[]byte("1.2"), 1.2},
		{[]byte("5.5"), 5.5},
		{[]byte("9.9"), 9.9},
		{[]byte("-1.2"), -1.2},
		{[]byte("-5.5"), -5.5},
		{[]byte("-9.9"), -9.9},
		{[]byte("13.2"), 13.2},
		{[]byte("56.5"), 56.5},
		{[]byte("98.9"), 98.9},
		{[]byte("-13.2"), -13.2},
		{[]byte("-56.5"), -56.5},
		{[]byte("-98.9"), -98.9},
	}
	for _, tc := range testCases {
		temp := ParseTemp(tc.input)
		if temp.AsFloat() != tc.expectedTemp {
			panic(fmt.Sprintf("expected temp %d, got %d", tc.expectedTemp, temp.AsFloat()))
		}
	}
}
