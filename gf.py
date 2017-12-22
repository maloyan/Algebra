def rm_zero(A):
	while(A[0]==0):
		if(len(A)>1):
			A.remove(0)
		else:
			break
	return A


def add(Ax, Bx):
	for x in xrange(len(Ax)-len(Bx)):
		Bx.insert(0,0)	
	Cx=[]
	for x in xrange(len(Ax)):
		Ci = Ax[x] ^ Bx[x]
		Cx.append(Ci)
	Cx = rm_zero(Cx)
	return Cx

def binary(A):
	Cx=[]
	while(A!=0):
		Cx.insert(0,A%2)
		A/=2
	return Cx
def decimal(A):
	exp = len(A)-1
	dec = 0
	for x in xrange(len(A)):
		dec +=A[x]*2**exp
		exp-=1
	return dec

def mod(A,Px):
	p = decimal(Px)
	binA = binary(A)
	shift = len(binA) - len(Px)
	while(len(binA)>=len(Px)):
		A = decimal(binA) ^ (p<<shift)
		shift = len(binA) - len(Px)
		binA=binary(A)
	return A

def binMult(A, B, Px):
	prod = []
	for x in xrange(len(B)-1, -1, -1):
		add = []
		for y in xrange(len(A)-1, -1, -1):
			add.insert(0,A[y]*B[x])
		# print add
		prod.append(decimal(add))
	for x in xrange(len(prod)):
		prod[x]=prod[x]<<x
	# print prod
	xor=0
	for x in xrange(len(prod)):
		if(x==0):
			xor=prod[x]
		else:
			xor^=prod[x]
	return mod(xor, Px)

def multiply(A, B, Px):
	prod = []
	for x in xrange(len(B)-1, -1, -1):
		add = []
		for y in xrange(len(A)-1, -1, -1):
			add.insert(0,binMult(binary(A[y]),binary(B[x]), Px))
		# print add
		prod.append(add)
	for x in xrange(len(prod)):
		prod[x]+=[0]*x
	max_len=0
	for x in xrange(len(prod)):
		if(len(prod[x])> max_len):
			max_len=len(prod[x])
	for x in xrange(len(prod)):
		# print "yes"
		insert = max_len - len(prod[x])
		# print insert
		for y in xrange(insert):
			prod[x].insert(0,0)
	# print prod
	answer = []
	for x in xrange(max_len):
		ans = 0
		for y in xrange(len(prod)):
			if(y==0):
				ans = prod[y][x]
			else:
				ans ^= prod[y][x]
		answer.append(ans)
	answer = rm_zero(answer)
	return answer

def divide(Ax, Bx, Px):
	ans = []
	remainder = 0
	while(True):
		for x in xrange(1,2**len(Px),1):
			div = multiply(Bx, [x], Px)
			# print div
			# print Ax
			if(div[0]==Ax[0]):
				insert = len(Ax) - len(div)
				div +=[0]*insert
				remainder = add(Ax,div)
				ans.append(x)
				break
		# print len(remainder), len(Bx)
		if(len(remainder)<len(Bx)):
			break
		else:
			Ax = remainder
	return ans, remainder

def printArray(A):
	string = ""
	for x in xrange(len(A)):
		string+=str(A[x])
	print string

Ax = raw_input("A(x): ")
Bx = raw_input("B(x): ")
Px = raw_input("P(x): ")

# Ax = "1 0 7 6"
# Bx = "1 6 3"
# Px = "1 0 1 1"

Ax = Ax.split()
Bx = Bx.split()
Px = Px.split()

valid=True

for x in xrange(len(Ax)):
	if(Ax[x].isdigit()==False):
		valid=False
		break
	Ax[x]=int(Ax[x])

for x in xrange(len(Bx)):
	if(Bx[x].isdigit()==False):
		valid=False
		break
	Bx[x]=int(Bx[x])

for x in xrange(len(Px)):
	if(Px[x].isdigit()==False):
		valid=False
		break
	Px[x]=int(Px[x])

for x in xrange(len(Ax)):
	if(Ax[x]>(2**(len(Px)-1)-1)):
		valid=False
		break

for x in xrange(len(Bx)):
	if(Bx[x]>(2**(len(Px)-1)-1)):
		valid=False
		break
	
if(valid==False):
	print "Invalid input"
	exit()

# print Px

print "[1] A(x) + B(x)\n[2] A(x) - B(x)\n[3] A(x) x B(x)\n[4] A(x) / B(x)"

choice = raw_input("Enter the number corresponding to your operation: ")
if(choice == "1" or choice == "2"):
	Cx = add(Ax, Bx)
	print "A(x) =",
	printArray(Ax)
	print "B(x) =",
	printArray(Bx)
	if(choice == "1"):
		print "A(x) + B(x) =",
		printArray(Cx)
	elif(choice == "2"):
		print "A(x) - B(x) =",
		printArray(Cx)
elif(choice == "3"):
	Cx = multiply(Ax, Bx, Px)
	print "A(x) =",
	printArray(Ax)
	print "B(x) =",
	printArray(Bx)
	print "A(x) x B(x) =",
	printArray(Cx)
elif(choice == "4"):
	Cx, remainder = divide(Ax, Bx, Px)
	print "A(x) =",
	printArray(Ax)
	print "B(x) =",
	printArray(Bx)
	print "A(x) / B(x) =",
	printArray(Cx)
	print "Remainder =",
	printArray(remainder)
