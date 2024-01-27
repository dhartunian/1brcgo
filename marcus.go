package main

import (
	"bufio"
	"fmt"
	"math"
	"math/bits"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"slices"
	"sync"
	"syscall"
	"unsafe"
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
		TestSemiColonPosition()
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
	size := uint64(info.Size())
	chunkSize := size / numGoroutines

	f, err := os.Open(inputFile)
	if err != nil {
		panic(err)
	}
	// mmap the file.
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		panic(err)
	}

	var m0, m1, m2, m3, m4, m5, m6, m7, m8, m9 *SummaryMap
	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		start := uint64(i) * chunkSize
		end := uint64(i+1) * chunkSize
		if i == numGoroutines-1 {
			end = uint64(len(data)) - 1
		}

		wg.Add(1)
		go func(g int, start, end uint64) {
			// Allocate a summary map.
			var m SummaryMap
			switch g {
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

			firstScanner := g == 0
			lastScanner := g == numGoroutines-1
			Scan(&m, data, start, end, firstScanner, lastScanner)

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

// Scan scans the data from start until ~end, adding records to the SummaryMap.
func Scan(m *SummaryMap, data []byte, start, end uint64, firstScanner, lastScanner bool) {
	head := start
	if !firstScanner {
		// Skip the first line unless this is the goroutine reading from
		// the beginning. The previous goroutine will scan the skipped
		// line.
		for ; head < end; head++ {
			if data[head] == '\n' {
				head++
				break
			}
		}
	}
	scanPast := end
	if lastScanner {
		// We can't scan past the end of the file.
		scanPast -= 2
	}
	for {
		var city []byte
		if head < 0 {
			return
		}

		i := head
		for ; i < uint64(len(data)); i += 8 {
			u := *(*uint64)(unsafe.Pointer(&data[i]))
			p := SemiColonPosition(u)
			if p >= 0 {
				city = data[head : i+uint64(p)]
				head = i + uint64(p) + 1
				break
			}
		}
		if city == nil {
			break
		}

		temp, adv := ParseTemp(data[head:])
		head += adv
		sum := m.Lookup(city)
		sum.Add(temp)

		if head > scanPast {
			break
		}
	}
}

const (
	shift1 = 8 * 1
	shift2 = 8 * 2
	shift3 = 8 * 3
	shift4 = 8 * 4

	charMask0 = int64(255)
	charMask1 = int64(255) << shift1
	charMask2 = int64(255) << shift2
	charMask3 = int64(255) << shift3
	charMask4 = int64(255) << shift4

	dot1 = int64('.') << 8
	dot2 = int64('.') << 16

	semiColonTest = 0x3B3B3B3B3B3B3B3B
)

// SemiColonPosition returns the position of the byte within u that represents a
// semi-colon character in ASCII. If such a byte is not present, it returns -1.
func SemiColonPosition(u uint64) int {
	// See "determine if a word has a byte equal to n", from "Bit Twiddling
	// Hacks",
	// https://graphics.stanford.edu/~seander/bithacks.html.
	b := u ^ semiColonTest
	b = (b - 0x0101010101010101) & (^b & 0x8080808080808080)
	if b == 0 {
		return -1
	}
	// The first bit of each byte in b that matches a semi-colon character will
	// be set. So we can count the trailing zeros (on a little-endian machine)
	// to find the position of the first semi-colon.
	z := bits.TrailingZeros64(b)
	// Divide by 8.
	return z >> 3
}

func ParseTemp(b []byte) (_ Temp, advance uint64) {
	// Gather the key and fetch the corresponding record. We can do this without
	// scanning the line because there are only four possible sequences for
	// the temperature. The valid formats are:
	//
	//     0.0
	//    00.0
	//    -0.0
	//   -00.0
	//
	// We use the limited locations of semicolons and minus characters to avoid
	// conditional expressions and loops.
	i := *(*int64)(unsafe.Pointer(&b[0]))
	switch {
	case (i & charMask1) == dot1:
		// Case: "0.0".
		ones := (i&charMask0 - '0') * 10
		tenths := i&charMask2>>shift2 - '0'
		// Advance past the newline, which is the fourth character.
		return Temp(ones + tenths), 4

	case (i & charMask2) == dot2:
		// Case: "00.0" and "-0.0".
		v0 := i & charMask0
		tens := (v0 - '0') * 100
		ones := (i&charMask1>>shift1 - '0') * 10
		tenths := i&charMask3>>shift3 - '0'

		// neg is 1 if the first character is the minus charater, and 0
		// otherwise.
		//
		// NOTE: The Go compiler eliminates jumps when using this form of
		// conditional.
		var neg int64
		if v0 == '-' {
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

		// Advance past the newline, which is the fifth character.
		return Temp(temp), 5
	default:
		// Case: "-00.0".
		tens := (i&charMask1>>shift1 - '0') * 100
		ones := (i&charMask2>>shift2 - '0') * 10
		tenths := i&charMask4>>shift4 - '0'

		t := tens + ones + tenths
		t *= -1
		// Advance past the newline, which is the sixth character.
		return Temp(t), 6
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

// /////////////////////////////////////////////////////////
// Tests
// /////////////////////////////////////////////////////////

func TestParseTemp() {
	type testCase struct {
		input           []byte
		expectedTemp    float64
		expectedAdvance uint64
	}
	testCases := []testCase{
		{[]byte("0.0"), 0, 4},
		{[]byte("0.1"), 0.1, 4},
		{[]byte("0.5"), 0.5, 4},
		{[]byte("0.9"), 0.9, 4},
		{[]byte("1.2"), 1.2, 4},
		{[]byte("5.5"), 5.5, 4},
		{[]byte("9.9"), 9.9, 4},
		{[]byte("-1.2"), -1.2, 5},
		{[]byte("-5.5"), -5.5, 5},
		{[]byte("-9.9"), -9.9, 5},
		{[]byte("13.2"), 13.2, 5},
		{[]byte("56.5"), 56.5, 5},
		{[]byte("98.9"), 98.9, 5},
		{[]byte("-13.2"), -13.2, 6},
		{[]byte("-56.5"), -56.5, 6},
		{[]byte("-98.9"), -98.9, 6},
	}
	for _, tc := range testCases {
		temp, adv := ParseTemp(tc.input)
		if temp.AsFloat() != tc.expectedTemp {
			panic(fmt.Sprintf("expected temp %d, got %d", tc.expectedTemp, temp.AsFloat()))
		}
		if adv != tc.expectedAdvance {
			panic(fmt.Sprintf("expected advance %d, got %d", tc.expectedAdvance, adv))
		}
	}
}

func TestSemiColonPosition() {
	type testCase struct {
		input    []byte
		expected int
	}
	testCases := []testCase{
		{[]byte("abcdefgh"), -1},
		{[]byte(";bcdefgh"), 0},
		{[]byte("a;cdefgh"), 1},
		{[]byte("a;cd;fgh"), 1},
		{[]byte("a;;;;;;;"), 1},
		{[]byte("ab;defgh"), 2},
		{[]byte("abc;efgh"), 3},
		{[]byte("abcd;fgh"), 4},
		{[]byte("abcde;gh"), 5},
		{[]byte("abcdef;h"), 6},
		{[]byte("abcdef;;"), 6},
		{[]byte("abcdefg;"), 7},
		{[]byte("abcdefgh;"), -1},
	}
	for _, tc := range testCases {
		i := *(*uint64)(unsafe.Pointer(&tc.input[0]))
		r := SemiColonPosition(i)
		if r != tc.expected {
			panic(fmt.Sprintf("%q: expected %d, got %d", tc.input, tc.expected, r))
		}
	}
}
