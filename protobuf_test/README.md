# Protobuf 학습/테스트

* python protobuf 설치
```
$ sudo apt install protobuf-compiler
```

* 설치 버전 확인
```
$ protoc --version
libprotoc 2.6.1
```

* Python 모듈 설치
```
$ pip3 install protobuf
```

* student.proto 작성

* protoc로 python 코드 생성, student_pb2.py 파일이 생성됨
```
$ protoc --python_out=. student.proto
```

* stduent 테스트, student_test.py 작성, student.dat 파일이 생성됨
```
$ python3 student_test.py
Name: Gildong Hong
StdNum: 2019-012345
Gender: MALE

Name: Junyoung Heo
StdNum: 2009-000005
Gender: MALE
Phone: 010-4401-0000
```

# ONNX 테스트

* https://github.com/onnx/onnx

* onnx python 설치
```
$ pip3 install onnx --user
```

* https://github.com/onnx/onnx/blob/master/onnx/examples/load_model.ipynb

* single_relu.onnx 파일을 load해서 print 해보기
```
$ python3 onnx_load.py
ir_version: 3
producer_name: "backend-test"
graph {
  node {
    input: "x"
    output: "y"
    name: "test"
    op_type: "Relu"
  }
  name: "SingleRelu"
  input {
    name: "x"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  output {
    name: "y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
}
opset_import {
  version: 6
}
```

* protoc로 python 코드 생성, onnx_pb2.py 파일이 생성됨
```
$ protoc --python_out=. onnx.proto
```

* onnx_test.py 로 single_relu.onnx 읽어보기
```
$ python3 onnx_test.py
ir_version: 3
graph.name: SingleRelu
[input: "x"
output: "y"
name: "test"
op_type: "Relu"
]
[name: "x"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 2
      }
    }
  }
}
]
[name: "y"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 2
      }
    }
  }
}
]
```
