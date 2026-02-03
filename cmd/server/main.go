package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"fan-ui/internal/config"
	"fan-ui/internal/router"
	"fan-ui/pkg/logger"
)

const defaultConfigPath = "configs/config.yaml"

func main() {
	cfg, err := config.Load(getConfigPath())
	if err != nil {
		panic(err)
	}

	log := logger.New(cfg.Log.Level).With(
		slog.String("app", cfg.App.Name),
		slog.String("env", cfg.App.Env),
	)

	server := &http.Server{
		Addr:    cfg.App.Address,
		Handler: router.New(log),
	}

	go func() {
		log.Info("server starting", slog.String("address", cfg.App.Address))
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server error", slog.Any("error", err))
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), cfg.App.ShutdownTimeout)
	defer cancel()

	log.Info("server shutting down")
	if err := server.Shutdown(ctx); err != nil {
		log.Error("server shutdown error", slog.Any("error", err))
	}

	<-ctx.Done()
	time.Sleep(50 * time.Millisecond)
	log.Info("server stopped")
}

func getConfigPath() string {
	if path := os.Getenv("APP_CONFIG"); path != "" {
		return path
	}
	return defaultConfigPath
}
