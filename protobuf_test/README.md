# Protobuf 학습/테스트

* python protobuf 설치
$ sudo apt install protobuf-compiler

* 설치 버전 확인
$ protoc --version
libprotoc 2.6.1

* Python 모듈 설치
$ pip3 install protobuf

* student.proto 작성

* protoc로 python 코드 생성
$ protoc --python_out=. student.proto
student_pb2.py 파일이 생성됨

* stduent 테스트, student_test.py 작성, student.dat 파일이 생성됨
$ python3 student_test.py
Name: Gildong Hong
StdNum: 2019-012345
Gender: MALE

Name: Junyoung Heo
StdNum: 2009-000005
Gender: MALE
Phone: 010-4401-0000


