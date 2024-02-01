package main

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

const maxCities = 10_000

// temp has exactly one fractional digit.
type temp int64

func (t temp) asFloat64() float64 {
	return float64(t) / 10
}

// record is a city's record.
type record struct {
	city  string
	min   temp
	max   temp
	sum   temp
	count int64
}

func (r *record) mean() temp {
	return temp(math.Round(float64(r.sum) / float64(r.count)))
}

// unsafeIdx returns data[i] without bounds checking.
func unsafeIdx[T any](dataPtr unsafe.Pointer, i int) T {
	// NOTE: this performs better than `return *unsafeIdxPtr[T](dataPtr, i)`.
	var v T
	return *(*T)(unsafe.Add(dataPtr, int(unsafe.Sizeof(v))*i))
}

// unsafeIdxPtr returns &data[i] without bounds checking.
func unsafeIdxPtr[T any](dataPtr unsafe.Pointer, i int) *T {
	var v T
	return (*T)(unsafe.Add(dataPtr, int(unsafe.Sizeof(v))*i))
}

// unsafeString returns string(data[i:j]) without bounds checking or copying.
func unsafeString(dataPtr unsafe.Pointer, i, j int) string {
	return unsafe.String((*byte)(unsafe.Add(dataPtr, i)), j-i)
}

// boolToInt converts a bool to an int. It is optimized by the go compiler to
// avoid branching.
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// fnv64a is a 64-bit FNV-1a hash.
type fnv64a uint64

// const from src/hash/fnv/fnv.go.
const (
	offset64 fnv64a = 14695981039346656037
	prime64  fnv64a = 1099511628359
)

func makeHash(b byte) fnv64a {
	return offset64.hashByte(b)
}

func (h fnv64a) hashByte(b byte) fnv64a {
	return (h ^ fnv64a(b)) * prime64
}

// simpleMap is a map from K to *V, using a linear probing open addressing
// scheme.
type simpleMap[K ~uint64, V any] struct {
	ptr unsafe.Pointer // []simpleMapSlot[K, V]
}

// simpleMapSlot is a slot in a simpleMap.
type simpleMapSlot[K ~uint64, V any] struct {
	k K
	v *V
}

// simpleMapSlots is the number of slots in a simpleMap. It is a power of two
// to allow for fast modulus operations.
const simpleMapSlots = 1 << 16 // 64K slots

func newSimpleMap[K ~uint64, V any]() simpleMap[K, V] {
	slots := make([]simpleMapSlot[K, V], simpleMapSlots)
	return simpleMap[K, V]{
		ptr: unsafe.Pointer(&slots[0]),
	}
}

func (m simpleMap[K, V]) slot(k K) int {
	return int(uint64(k) & (simpleMapSlots - 1))
}

func (m simpleMap[K, V]) nextSlot(i int) int {
	return (i + 1) & (simpleMapSlots - 1)
}

func (m simpleMap[K, V]) get(k K) *V {
	i := m.slot(k)
	for {
		s := unsafeIdx[simpleMapSlot[K, V]](m.ptr, i)
		if s.k == k {
			return s.v
		}
		if s.k == 0 {
			return nil
		}
		i = m.nextSlot(i)
	}
}

func (m simpleMap[K, V]) set(k K, v *V) {
	i := m.slot(k)
	for {
		s := unsafeIdxPtr[simpleMapSlot[K, V]](m.ptr, i)
		if s.k == 0 {
			*s = simpleMapSlot[K, V]{k, v}
			return
		}
		i = m.nextSlot(i)
	}
}

func (m simpleMap[K, V]) iter(f func(k K, v *V)) {
	for i := 0; i < simpleMapSlots; i++ {
		s := unsafeIdx[simpleMapSlot[K, V]](m.ptr, i)
		if s.k != 0 {
			f(s.k, s.v)
		}
	}
}

// simpleAlloc is a T allocator.
type simpleAlloc[T any] struct {
	buf []T
	idx int
}

func newSimpleAlloc[T any](n int) simpleAlloc[T] {
	return simpleAlloc[T]{
		buf: make([]T, n),
	}
}

// new returns a new, empty T.
func (a *simpleAlloc[T]) new() *T {
	v := &a.buf[a.idx]
	a.idx++
	return v
}

// worker is a worker goroutine that operates on a slice of the input data.
type worker struct {
	data []byte
	res  simpleMap[fnv64a, record]
}

func (w *worker) run(wg *sync.WaitGroup) {
	defer wg.Done()
	dataPtr := unsafe.Pointer(&w.data[0])
	dataLen := len(w.data)
	temps := newSimpleMap[fnv64a, record]()
	alloc := newSimpleAlloc[record](maxCities)
	start, end := 0, 0
	for {
		// Parse city name while hashing.
		b := unsafeIdx[byte](dataPtr, start)
		hash := makeHash(b)
		end = start + 1
		for {
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			// Unrolled 15 more times...
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == ';' {
				break
			}
			hash = hash.hashByte(b)
			end++
		}
		cityEnd := end

		// Parse temperature.
		var t temp
		// Byte 1.
		sign := temp(+1)
		end++
		b = unsafeIdx[byte](dataPtr, end)
		if b == '-' {
			// If negative, move to byte 2.
			sign = -1
			end++
			b = unsafeIdx[byte](dataPtr, end)
		}
		t = temp(b - '0')
		// Byte 2 or 3, depending on sign.
		end++
		b = unsafeIdx[byte](dataPtr, end)
		// twoDigits is 1 if two digits before decimal, 0 if one digit.
		twoDigits := boolToInt(b != '.')
		t += temp(twoDigits) * (t*9 + temp(b-'0'))
		end += twoDigits
		// Skip past decimal.
		end++
		// Fractional digit.
		b = unsafeIdx[byte](dataPtr, end)
		t = t*10 + temp(b-'0')
		// Optionally negate.
		t = sign * t

		// Get city's record.
		r := temps.get(hash)
		if r == nil {
			r = alloc.new()
			r.city = unsafeString(dataPtr, start, cityEnd)
			r.min = math.MaxInt64
			r.max = math.MinInt64
			temps.set(hash, r)
		}

		// Update city's record.
		r.min = min(r.min, t)
		r.max = max(r.max, t)
		r.sum = r.sum + t
		r.count = r.count + 1

		// Iterate to next line.
		start = end + 2
		if start >= dataLen {
			break
		}

		// Unrolled 7 more times...
		{
			// Parse city name while hashing.
			b := unsafeIdx[byte](dataPtr, start)
			hash := makeHash(b)
			end = start + 1
			for {
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				// Unrolled 15 more times...
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
			}
			cityEnd := end

			// Parse temperature.
			var t temp
			// Byte 1.
			sign := temp(+1)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == '-' {
				// If negative, move to byte 2.
				sign = -1
				end++
				b = unsafeIdx[byte](dataPtr, end)
			}
			t = temp(b - '0')
			// Byte 2 or 3, depending on sign.
			end++
			b = unsafeIdx[byte](dataPtr, end)
			// twoDigits is 1 if two digits before decimal, 0 if one digit.
			twoDigits := boolToInt(b != '.')
			t += temp(twoDigits) * (t*9 + temp(b-'0'))
			end += twoDigits
			// Skip past decimal.
			end++
			// Fractional digit.
			b = unsafeIdx[byte](dataPtr, end)
			t = t*10 + temp(b-'0')
			// Optionally negate.
			t = sign * t

			// Get city's record.
			r := temps.get(hash)
			if r == nil {
				r = alloc.new()
				r.city = unsafeString(dataPtr, start, cityEnd)
				r.min = math.MaxInt64
				r.max = math.MinInt64
				temps.set(hash, r)
			}

			// Update city's record.
			r.min = min(r.min, t)
			r.max = max(r.max, t)
			r.sum = r.sum + t
			r.count = r.count + 1

			// Iterate to next line.
			start = end + 2
			if start >= dataLen {
				break
			}
		}
		{
			// Parse city name while hashing.
			b := unsafeIdx[byte](dataPtr, start)
			hash := makeHash(b)
			end = start + 1
			for {
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				// Unrolled 15 more times...
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
			}
			cityEnd := end

			// Parse temperature.
			var t temp
			// Byte 1.
			sign := temp(+1)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == '-' {
				// If negative, move to byte 2.
				sign = -1
				end++
				b = unsafeIdx[byte](dataPtr, end)
			}
			t = temp(b - '0')
			// Byte 2 or 3, depending on sign.
			end++
			b = unsafeIdx[byte](dataPtr, end)
			// twoDigits is 1 if two digits before decimal, 0 if one digit.
			twoDigits := boolToInt(b != '.')
			t += temp(twoDigits) * (t*9 + temp(b-'0'))
			end += twoDigits
			// Skip past decimal.
			end++
			// Fractional digit.
			b = unsafeIdx[byte](dataPtr, end)
			t = t*10 + temp(b-'0')
			// Optionally negate.
			t = sign * t

			// Get city's record.
			r := temps.get(hash)
			if r == nil {
				r = alloc.new()
				r.city = unsafeString(dataPtr, start, cityEnd)
				r.min = math.MaxInt64
				r.max = math.MinInt64
				temps.set(hash, r)
			}

			// Update city's record.
			r.min = min(r.min, t)
			r.max = max(r.max, t)
			r.sum = r.sum + t
			r.count = r.count + 1

			// Iterate to next line.
			start = end + 2
			if start >= dataLen {
				break
			}
		}
		{
			// Parse city name while hashing.
			b := unsafeIdx[byte](dataPtr, start)
			hash := makeHash(b)
			end = start + 1
			for {
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				// Unrolled 15 more times...
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
			}
			cityEnd := end

			// Parse temperature.
			var t temp
			// Byte 1.
			sign := temp(+1)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == '-' {
				// If negative, move to byte 2.
				sign = -1
				end++
				b = unsafeIdx[byte](dataPtr, end)
			}
			t = temp(b - '0')
			// Byte 2 or 3, depending on sign.
			end++
			b = unsafeIdx[byte](dataPtr, end)
			// twoDigits is 1 if two digits before decimal, 0 if one digit.
			twoDigits := boolToInt(b != '.')
			t += temp(twoDigits) * (t*9 + temp(b-'0'))
			end += twoDigits
			// Skip past decimal.
			end++
			// Fractional digit.
			b = unsafeIdx[byte](dataPtr, end)
			t = t*10 + temp(b-'0')
			// Optionally negate.
			t = sign * t

			// Get city's record.
			r := temps.get(hash)
			if r == nil {
				r = alloc.new()
				r.city = unsafeString(dataPtr, start, cityEnd)
				r.min = math.MaxInt64
				r.max = math.MinInt64
				temps.set(hash, r)
			}

			// Update city's record.
			r.min = min(r.min, t)
			r.max = max(r.max, t)
			r.sum = r.sum + t
			r.count = r.count + 1

			// Iterate to next line.
			start = end + 2
			if start >= dataLen {
				break
			}
		}
		{
			// Parse city name while hashing.
			b := unsafeIdx[byte](dataPtr, start)
			hash := makeHash(b)
			end = start + 1
			for {
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				// Unrolled 15 more times...
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
			}
			cityEnd := end

			// Parse temperature.
			var t temp
			// Byte 1.
			sign := temp(+1)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == '-' {
				// If negative, move to byte 2.
				sign = -1
				end++
				b = unsafeIdx[byte](dataPtr, end)
			}
			t = temp(b - '0')
			// Byte 2 or 3, depending on sign.
			end++
			b = unsafeIdx[byte](dataPtr, end)
			// twoDigits is 1 if two digits before decimal, 0 if one digit.
			twoDigits := boolToInt(b != '.')
			t += temp(twoDigits) * (t*9 + temp(b-'0'))
			end += twoDigits
			// Skip past decimal.
			end++
			// Fractional digit.
			b = unsafeIdx[byte](dataPtr, end)
			t = t*10 + temp(b-'0')
			// Optionally negate.
			t = sign * t

			// Get city's record.
			r := temps.get(hash)
			if r == nil {
				r = alloc.new()
				r.city = unsafeString(dataPtr, start, cityEnd)
				r.min = math.MaxInt64
				r.max = math.MinInt64
				temps.set(hash, r)
			}

			// Update city's record.
			r.min = min(r.min, t)
			r.max = max(r.max, t)
			r.sum = r.sum + t
			r.count = r.count + 1

			// Iterate to next line.
			start = end + 2
			if start >= dataLen {
				break
			}
		}
		{
			// Parse city name while hashing.
			b := unsafeIdx[byte](dataPtr, start)
			hash := makeHash(b)
			end = start + 1
			for {
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				// Unrolled 15 more times...
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
			}
			cityEnd := end

			// Parse temperature.
			var t temp
			// Byte 1.
			sign := temp(+1)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == '-' {
				// If negative, move to byte 2.
				sign = -1
				end++
				b = unsafeIdx[byte](dataPtr, end)
			}
			t = temp(b - '0')
			// Byte 2 or 3, depending on sign.
			end++
			b = unsafeIdx[byte](dataPtr, end)
			// twoDigits is 1 if two digits before decimal, 0 if one digit.
			twoDigits := boolToInt(b != '.')
			t += temp(twoDigits) * (t*9 + temp(b-'0'))
			end += twoDigits
			// Skip past decimal.
			end++
			// Fractional digit.
			b = unsafeIdx[byte](dataPtr, end)
			t = t*10 + temp(b-'0')
			// Optionally negate.
			t = sign * t

			// Get city's record.
			r := temps.get(hash)
			if r == nil {
				r = alloc.new()
				r.city = unsafeString(dataPtr, start, cityEnd)
				r.min = math.MaxInt64
				r.max = math.MinInt64
				temps.set(hash, r)
			}

			// Update city's record.
			r.min = min(r.min, t)
			r.max = max(r.max, t)
			r.sum = r.sum + t
			r.count = r.count + 1

			// Iterate to next line.
			start = end + 2
			if start >= dataLen {
				break
			}
		}
		{
			// Parse city name while hashing.
			b := unsafeIdx[byte](dataPtr, start)
			hash := makeHash(b)
			end = start + 1
			for {
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				// Unrolled 15 more times...
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
			}
			cityEnd := end

			// Parse temperature.
			var t temp
			// Byte 1.
			sign := temp(+1)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == '-' {
				// If negative, move to byte 2.
				sign = -1
				end++
				b = unsafeIdx[byte](dataPtr, end)
			}
			t = temp(b - '0')
			// Byte 2 or 3, depending on sign.
			end++
			b = unsafeIdx[byte](dataPtr, end)
			// twoDigits is 1 if two digits before decimal, 0 if one digit.
			twoDigits := boolToInt(b != '.')
			t += temp(twoDigits) * (t*9 + temp(b-'0'))
			end += twoDigits
			// Skip past decimal.
			end++
			// Fractional digit.
			b = unsafeIdx[byte](dataPtr, end)
			t = t*10 + temp(b-'0')
			// Optionally negate.
			t = sign * t

			// Get city's record.
			r := temps.get(hash)
			if r == nil {
				r = alloc.new()
				r.city = unsafeString(dataPtr, start, cityEnd)
				r.min = math.MaxInt64
				r.max = math.MinInt64
				temps.set(hash, r)
			}

			// Update city's record.
			r.min = min(r.min, t)
			r.max = max(r.max, t)
			r.sum = r.sum + t
			r.count = r.count + 1

			// Iterate to next line.
			start = end + 2
			if start >= dataLen {
				break
			}
		}
		{
			// Parse city name while hashing.
			b := unsafeIdx[byte](dataPtr, start)
			hash := makeHash(b)
			end = start + 1
			for {
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				// Unrolled 15 more times...
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
				b = unsafeIdx[byte](dataPtr, end)
				if b == ';' {
					break
				}
				hash = hash.hashByte(b)
				end++
			}
			cityEnd := end

			// Parse temperature.
			var t temp
			// Byte 1.
			sign := temp(+1)
			end++
			b = unsafeIdx[byte](dataPtr, end)
			if b == '-' {
				// If negative, move to byte 2.
				sign = -1
				end++
				b = unsafeIdx[byte](dataPtr, end)
			}
			t = temp(b - '0')
			// Byte 2 or 3, depending on sign.
			end++
			b = unsafeIdx[byte](dataPtr, end)
			// twoDigits is 1 if two digits before decimal, 0 if one digit.
			twoDigits := boolToInt(b != '.')
			t += temp(twoDigits) * (t*9 + temp(b-'0'))
			end += twoDigits
			// Skip past decimal.
			end++
			// Fractional digit.
			b = unsafeIdx[byte](dataPtr, end)
			t = t*10 + temp(b-'0')
			// Optionally negate.
			t = sign * t

			// Get city's record.
			r := temps.get(hash)
			if r == nil {
				r = alloc.new()
				r.city = unsafeString(dataPtr, start, cityEnd)
				r.min = math.MaxInt64
				r.max = math.MinInt64
				temps.set(hash, r)
			}

			// Update city's record.
			r.min = min(r.min, t)
			r.max = max(r.max, t)
			r.sum = r.sum + t
			r.count = r.count + 1

			// Iterate to next line.
			start = end + 2
			if start >= dataLen {
				break
			}
		}
	}
	w.res = temps
}

func main() {
	// Disable the GC. It shouldn't need to run, but make sure it doesn't.
	debug.SetGCPercent(-1)

	f, data, err := mmap("measurements.txt")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// We use more threads than there are CPUs because doing so seems to
	// smooth out variation in the performance of any single thread. This
	// may also permit more parallelism during page faults — I'm not sure.
	const threadsPerCPU = 4
	numCpus := runtime.NumCPU()
	numWorkers := threadsPerCPU * numCpus
	runtime.GOMAXPROCS(numWorkers)

	// Compute start indices for each worker.
	stride := len(data) / numWorkers
	starts := make([]int, numWorkers)
	for i := range starts {
		if i == 0 {
			continue
		}
		start := i * stride
		for data[start-1] != '\n' {
			start++
		}
		starts[i] = start
	}

	// Create workers.
	workers := make([]*worker, numWorkers)
	for i := range workers {
		start := starts[i]
		var end int
		if i == numWorkers-1 {
			end = len(data)
		} else {
			end = starts[i+1]
		}
		workers[i] = &worker{
			data: data[start:end],
		}
	}

	// Launch workers.
	var wg sync.WaitGroup
	wg.Add(len(workers))
	for _, w := range workers[1:] {
		go w.run(&wg)
	}
	workers[0].run(&wg)
	wg.Wait()

	// Merge results.
	var temps simpleMap[fnv64a, record]
	for i, w := range workers {
		if i == 0 {
			temps = w.res
			continue
		}
		w.res.iter(func(k fnv64a, r2 *record) {
			r := temps.get(k)
			if r == nil {
				temps.set(k, r2)
				return
			}
			r.min = min(r.min, r2.min)
			r.max = max(r.max, r2.max)
			r.sum = r.sum + r2.sum
			r.count = r.count + r2.count
		})
	}

	// Sort results.
	byName := make([]*record, 0, maxCities)
	temps.iter(func(_ fnv64a, r *record) {
		byName = append(byName, r)
	})
	slices.SortFunc(byName, func(a, b *record) int {
		return strings.Compare(a.city, b.city)
	})

	// Print results.
	for _, r := range byName {
		fmt.Printf("%s=%.1f/%.1f/%.1f\n", r.city, r.min.asFloat64(), r.mean().asFloat64(), r.max.asFloat64())
	}
}

func mmap(filename string) (_ *os.File, data []byte, err error) {
	// Open the file.
	f, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()

	// Grab its size.
	fi, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}

	// mmap the file.
	data, err = syscall.Mmap(int(f.Fd()), 0, int(fi.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return f, data, nil
}
