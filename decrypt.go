package main

import (
	"flag"
	"os/exec"
)

func main() {
	flag.Parse()
	fileName := flag.Arg(0)
	encFileName := fileName + ".enc"
	keyFileName := fileName + ".key"

	// Decrypt the symmetric key.
	execCmd("openssl", "rsautl", "-decrypt", "-inkey", "private_key.pem", "-in", keyFileName, "-out", "key.sym")
	defer execCmd("rm", "key.sym")

	// Decrypt the file.
	execCmd("openssl", "aes-256-cbc", "-d", "-a", "-pbkdf2", "-in", encFileName, "-out", fileName, "-kfile", "key.sym")
}

func execCmd(name string, args ...string) {
	cmd := exec.Command(name, args...)
	if err := cmd.Run(); err != nil {
		panic(err)
	}
}
