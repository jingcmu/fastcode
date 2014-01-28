PROJ = matrix_mul
CC = g++

CFLAGS = -c -g -Wall -I/opt/local/include -I$(HOME)/cppunit/include
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
LIBS = -lcppunit -ldl
OBJS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OMPFLAGS = -fopenmp
SIMDFLAGS = -msse2 -msse3 -O2 -lm

all: $(PROJ)

$(PROJ): $(OBJS)
	$(CC) $(OMPFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@ $(SIMDFLAGS) $(OMPFLAGS)

%.o : %.cpp %.h
	$(CC) $(CFLAGS) $< -o $@ $(SIMDFLAGS) $(OMPFLAGS)

clean:
	rm -f $(PROJ) $(OBJS) *.xml
