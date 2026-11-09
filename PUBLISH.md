# Publish GoFish

The module is prepared for `github.com/JWset/gofish`. It is not published merely
by creating the local Git repository. Run these commands from a terminal with
working GitHub network access and your own GitHub login:

```sh
cd /home/jwset/GolandProjects/gofish
gh auth login
gh repo create JWset/gofish --public --source=. --remote=origin --push \
  --description "PyTorch-inspired tensors, autograd, neural networks and pretrained text generation for Go"
```

After the repository is created and pushed, other projects can use:

```sh
go get github.com/JWset/gofish
```

For an initial tagged release:

```sh
git tag v0.1.0
git push origin v0.1.0
```

If you publish to a different owner or repository name, update `go.mod` and the
internal Go import paths to match that location before pushing.
