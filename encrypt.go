package main

import (
	"flag"
	"fmt"
	"os/exec"
)

func main() {
	flag.Parse()
	fileName := flag.Arg(0)

	// Create a symmetric key.
	execCmd("openssl", "rand", "-out", "key.sym", "-base64", "32")
	defer execCmd("rm", "key.sym")

	// Encrypt fileName with the symmetric key.
	execCmd("openssl", "aes-256-cbc", "-a", "-pbkdf2", "-salt", "-in", fileName, "-out", fileName+".enc", "-kfile", "key.sym")

	// Encrypt the symmetric key with the public key.
	execCmd("openssl", "rsautl", "-encrypt", "-pubin", "-inkey", "public_key.pem", "-in", "key.sym", "-out", fileName+".key")

	fmt.Printf("Created %s.enc and %s.key. Please send both.\n", fileName, fileName)
}

func execCmd(name string, args ...string) {
	cmd := exec.Command(name, args...)
	if err := cmd.Run(); err != nil {
		panic(err)
	}
}
