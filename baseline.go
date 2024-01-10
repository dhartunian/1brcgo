package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
)

type record struct {
	min   float64
	max   float64
	sum   float64
	count float64
}

func main() {
	f, err := os.Open("measurements.txt")
	if err != nil {
		panic(err)
	}
	defer func() {
		err := f.Close()
		if err != nil {
			panic(err)
		}
	}()

	temps := make(map[string]*record)
	var cities []string

	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		cityAndTemp := strings.Split(line, ";")
		if temps[cityAndTemp[0]] == nil {
			temps[cityAndTemp[0]] = &record{
				min: math.MaxFloat64,
				max: -math.MaxFloat64,
			}
			cities = append(cities, cityAndTemp[0])
		}
		i, err := strconv.ParseFloat(cityAndTemp[1], 10)
		if err != nil {
			panic(err)
		}
		r := temps[cityAndTemp[0]]
		r.min = min(r.min, i)
		r.max = max(r.max, i)
		r.sum = r.sum + i
		r.count = r.count + 1
	}

	slices.Sort(cities)
	for _, city := range cities {
		record := temps[city]
		fmt.Printf("%s=%.1f/%.1f/%.1f\n", city, record.min, record.sum/record.count, record.max)
	}
}
