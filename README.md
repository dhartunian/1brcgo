# Billion Row Challenge in go

Inspired by https://github.com/gunnarmorling/1brc

# Rules

* No external library dependencies may be used
* Implementations must be provided as a single source file
* The computation must happen at application runtime, i.e. you cannot process 
  the measurements file at build time (for instance, when using GraalVM) and just bake the result into the binary
* Input value ranges are as follows:
* Station name: non null UTF-8 string of min length 1 character and max 
  length 100 characters
* Temperature value: non null double between -99.9 (inclusive) and 99.9 
  (inclusive), always with one fractional digit
* There is a maximum of 10,000 unique station names
* Implementations must not rely on specifics of a given data set, e.g. any 
  valid station name as per the constraints above and any data distribution (number of measurements per station) must be supported