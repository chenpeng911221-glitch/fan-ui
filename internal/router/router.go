package router

import (
	"log/slog"

	"github.com/gin-gonic/gin"

	"fan-ui/internal/handler"
	"fan-ui/internal/middleware"
	"fan-ui/internal/service"
)

func New(log *slog.Logger) *gin.Engine {
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.Use(middleware.Logger(log))
	engine.Use(middleware.Recovery(log))

	healthHandler := handler.NewHealthHandler()
	engine.GET("/healthz", healthHandler.Healthz)
	engine.GET("/readyz", healthHandler.Readyz)

	userService := service.NewUserService()
	userHandler := handler.NewUserHandler(userService)

	api := engine.Group("/api/v1")
	api.GET("/users/:id", userHandler.GetUser)

	return engine
}
