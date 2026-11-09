// Command gofish generates text, trains native models and runs pretrained models.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/JWset/gofish"
	"github.com/JWset/gofish/ollama"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	if err := run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(ctx context.Context, args []string, in io.Reader, out, diagnostics io.Writer) error {
	if len(args) == 0 {
		fmt.Fprintln(out, "GoFish: generate | train | chat | pull | models\nTry: gofish generate -prompt 'the sun was'")
		return nil
	}
	if args[0] == "models" {
		for _, m := range gofish.Models() {
			fmt.Fprintf(out, "%-22s %s\n", m.Name, m.Description)
		}
		return nil
	}
	flags := flag.NewFlagSet("gofish "+args[0], flag.ContinueOnError)
	flags.SetOutput(diagnostics)
	modelName := flags.String("model", "tiny-stories", "bundled name, checkpoint path, or ollama:<name>")
	prompt := flags.String("prompt", "the sun was", "input text (native models require characters in their vocabulary)")
	tokens := flags.Int("tokens", 128, "maximum output tokens (characters for native models)")
	temperature := flags.Float64("temperature", 0.8, "sampling temperature; 0 is greedy")
	seed := flags.Int64("seed", 42, "random seed")
	dataPath := flags.String("data", "", "UTF-8 training corpus")
	outputPath := flags.String("out", "model.gofish", "native checkpoint destination")
	steps := flags.Int("steps", 500, "number of Adam updates")
	batch := flags.Int("batch", 32, "training batch size")
	rate := flags.Float64("lr", 0.01, "learning rate")
	contextSize := flags.Int("context", 12, "context characters for a new native model")
	embedding := flags.Int("embedding", 8, "embedding width for a new native model")
	hidden := flags.Int("hidden", 64, "hidden width for a new native model")
	host := flags.String("host", "http://localhost:11434", "Ollama server URL")
	timeout := flags.Duration("timeout", 10*time.Minute, "operation timeout")
	if err := flags.Parse(args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if flags.NArg() > 0 {
		return fmt.Errorf("gofish: unexpected arguments: %s", strings.Join(flags.Args(), " "))
	}
	if args[0] != "generate" && args[0] != "train" && args[0] != "chat" && args[0] != "pull" {
		return fmt.Errorf("gofish: unknown command %q", args[0])
	}
	if *timeout <= 0 {
		return fmt.Errorf("gofish: timeout must be positive")
	}
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	var model *gofish.Model
	var err error
	if args[0] == "train" && *modelName == "new" {
		if *dataPath == "" {
			return fmt.Errorf("gofish: train requires -data <file>")
		}
		data, readErr := os.ReadFile(*dataPath)
		if readErr != nil {
			return readErr
		}
		model, err = gofish.NewLanguageModel(string(data), gofish.Config{ContextSize: *contextSize, EmbeddingSize: *embedding, HiddenSize: *hidden, Seed: *seed})
	} else if strings.HasPrefix(*modelName, "ollama:") {
		model, err = gofish.Ollama(strings.TrimPrefix(*modelName, "ollama:"), ollama.WithBaseURL(*host))
	} else {
		model, err = gofish.Load(*modelName)
	}
	if err != nil {
		return err
	}
	options := []gofish.GenerateOption{gofish.WithMaxTokens(*tokens), gofish.WithTemperature(*temperature), gofish.WithSeed(*seed)}
	switch args[0] {
	case "generate":
		options = append(options, gofish.WithStream(func(s string) error { _, err := io.WriteString(out, s); return err }))
		_, err = model.Generate(ctx, *prompt, options...)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(out)
		return err
	case "pull":
		if err = model.Pull(ctx); err != nil {
			return err
		}
		fmt.Fprintf(out, "Ready: %s\n", *modelName)
		return nil
	case "train":
		if *dataPath == "" {
			return fmt.Errorf("gofish: train requires -data <file>")
		}
		data, err := os.ReadFile(*dataPath)
		if err != nil {
			return err
		}
		_, err = model.Train(ctx, string(data), gofish.TrainOptions{Steps: *steps, BatchSize: *batch, LearningRate: *rate, Seed: *seed, OnStep: func(s gofish.Step) {
			if s.Number == 1 || s.Number%100 == 0 || s.Number == *steps {
				fmt.Fprintf(diagnostics, "step=%d loss=%.4f\n", s.Number, s.Loss)
			}
		}})
		if err != nil {
			return err
		}
		if err = model.Save(*outputPath); err != nil {
			return err
		}
		fmt.Fprintf(out, "Saved %s\n", *outputPath)
		return nil
	case "chat":
		if !model.Capabilities().Chat {
			return fmt.Errorf("gofish: chat requires -model ollama:<name>; use generate for bundled models")
		}
		scanner := bufio.NewScanner(in)
		scanner.Buffer(make([]byte, 4096), 1<<20)
		history := []gofish.Message{}
		fmt.Fprintln(out, "GoFish chat. Type /quit to exit.")
		for {
			fmt.Fprint(out, "> ")
			if !scanner.Scan() {
				break
			}
			text := scanner.Text()
			if text == "/quit" {
				break
			}
			if strings.TrimSpace(text) == "" {
				continue
			}
			history = append(history, gofish.Message{Role: "user", Content: text})
			streamOptions := append(append([]gofish.GenerateOption(nil), options...), gofish.WithStream(func(s string) error { _, err := io.WriteString(out, s); return err }))
			reply, err := model.Chat(ctx, history, streamOptions...)
			if err != nil {
				return err
			}
			fmt.Fprintln(out)
			history = append(history, gofish.Message{Role: "assistant", Content: reply})
		}
		return scanner.Err()
	}
	return nil
}
