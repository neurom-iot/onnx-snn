import student_pb2

# make a student list
studentList = student_pb2.StudentList()
student = studentList.students.add()
student.name = 'Gildong Hong'
student.std_num.year = 2019
student.std_num.serial = "012345"
student.gender = student_pb2.Student.MALE

student = studentList.students.add()
student.name = 'Junyoung Heo'
student.std_num.year = 2009
student.std_num.serial = "000005"
student.gender = student_pb2.Student.MALE
student.phone = '010-4401-0000'
# write the list to the student.dat file
with open('student.dat', 'wb') as f:
    f.write(studentList.SerializeToString())

# read students list from student.dat file
studentList2 = student_pb2.StudentList()
with open('student.dat', 'rb') as f:
    studentList2.ParseFromString(f.read()) # parsing

# print(studentList2)

for student in studentList2.students:
    print('Name: {}'.format(student.name))
    print('StdNum: {}-{}'.format(student.std_num.year, student.std_num.serial))
    print('Gender: {}'.format('MALE' if student.gender == student_pb2.Student.MALE else 'FEMALE'))
    if student.HasField('phone'):
        print('Phone: {}'.format(student.phone))
    print()
