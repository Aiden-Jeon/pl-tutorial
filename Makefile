SHELL := /bin/bash

all:
	echo 'Makefile for pl-tutorial'

init:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .
