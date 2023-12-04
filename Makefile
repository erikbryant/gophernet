fmt:
	go fmt ./...

vet: fmt
	go vet ./...

test: vet
	go test ./...

run: test
	go build
	./gophernet

profile: test
	go build
	./gophernet -cpuprofile cpu.prof
	echo top | go tool pprof cpu.prof

# Targets that do not represent actual files
.PHONY: fmt test vet run profile
