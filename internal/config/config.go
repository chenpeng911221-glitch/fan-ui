package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type AppConfig struct {
	Name            string        `yaml:"name"`
	Env             string        `yaml:"env"`
	Address         string        `yaml:"address"`
	ShutdownTimeout time.Duration `yaml:"shutdown_timeout"`
}

type LogConfig struct {
	Level string `yaml:"level"`
}

type Config struct {
	App AppConfig `yaml:"app"`
	Log LogConfig `yaml:"log"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if cfg.App.Address == "" {
		cfg.App.Address = ":8080"
	}
	if cfg.App.ShutdownTimeout == 0 {
		cfg.App.ShutdownTimeout = 10 * time.Second
	}
	if cfg.App.Env == "" {
		cfg.App.Env = "dev"
	}
	if cfg.App.Name == "" {
		cfg.App.Name = "fan-ui"
	}
	if cfg.Log.Level == "" {
		cfg.Log.Level = "info"
	}

	return &cfg, nil
}
