package service

import (
	"context"
	"errors"

	"fan-ui/internal/model"
)

var ErrUserNotFound = errors.New("user not found")

type UserService struct{}

func NewUserService() *UserService {
	return &UserService{}
}

func (s *UserService) GetByID(_ context.Context, id string) (*model.User, error) {
	if id == "0" {
		return nil, ErrUserNotFound
	}
	return &model.User{
		ID:    id,
		Name:  "Gin User",
		Email: "user@example.com",
	}, nil
}
