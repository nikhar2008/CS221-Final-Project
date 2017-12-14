import random
import collections
import math
import sys

def isValidRoutingNumber(routNum):
	if routNum.len()!= 9: return False
	if not routNum.isDigit(): return False
	sumValFirst = 3*(int(routNum[0])+int(routNum[3])+int(routNum[6]))
	sumValSec = 7*(int(routNum[1]) + int(routNum[4])+ int(routNum[7]))
	sumValThird = int(routNum[2])+ int(routNum[5])+int(routNum[8])
	if (sumValThird+sumValSec+sumValFirst)%10 == 0: return True

	return False

def main():
	print ":::"
	print isValidRoutingNumber("1107380")
	print isValidRoutingNumber("123456789")
	print isValidRoutingNumber("021000021")
	print isValidRoutingNumber("011401533")
	print isValidRoutingNumber("011401yq3")
	print isValidRoutingNumber("011401iuhiu3")

if __name__ == "__main__": main()



