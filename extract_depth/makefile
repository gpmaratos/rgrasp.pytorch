python_path = /usr/include/python3.6m/
numpy_path = /usr/lib/python3.6/dist-packages/numpy/core/include/

read_pcd: read_pcd.cpp
	g++ -Wall -o read_pcd read_pcd.cpp -I $(python_path) -I $(numpy_path) -l python3.6m

install:
	python3 setup.py build
	python3 setup.py install --user

clean:
	rm -f read_pcd
