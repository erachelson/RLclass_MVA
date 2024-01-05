
all:
	mkdocs build

sync:
	cp docs/index.md README.md

deploy:
	mkdocs gh-deploy

clean:
	rm -rf site
