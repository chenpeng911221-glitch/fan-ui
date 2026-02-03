package middleware

import (
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"
)

func Recovery(log *slog.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if rec := recover(); rec != nil {
				log.Error("panic recovered", slog.Any("error", rec))
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"message": "internal server error"})
			}
		}()
		c.Next()
	}
}
