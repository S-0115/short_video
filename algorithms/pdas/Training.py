#-*- coding: UTF-8 -*-
import random
import string
import math
import os

segmentDuration = 2
# VrateList = [200, 800, 2200, 5000, 12000, 25000]
VrateList = [25000, 12000, 5000, 2200, 800, 200]
bufferLevelList = range(0, 10000, 500)
throughputList=range(1,50000,200)#1s的BW，Kbps

def downloadTimeCalculation(throughput,bitrate):
	"""
	本方法旨在计算出给定throughput,以bitrate下载segment所需时间
	:param throughput: 吞吐量
	:param bitrate: 比特率
	:return: 下载所需时间
	"""

	download_time = round(float(bitrate*segmentDuration)/throughput,1)
	download_time_int = int(bitrate*segmentDuration/throughput)


	if download_time==download_time_int:
		return download_time_int
	else:
		return download_time_int+1



def simulatePlay(throughput, preBitrate, bufferLevel, horizonRateList):
	'''
	本方法旨在计算出给定throughput, preBitrate, bufferLevel,计算horizonRateList比特率序列时的QoE
	:param throughput: 吞吐量
	:param preBitrate: 上一个segment的比特率
	:param bufferLevel: 缓冲区剩余量
	:param horizonRateList: 视野高度，即之后的比特率序列
	:return: QoE
	'''
	import math

	rebufferDuration=0
	# print(throughput, preBitrate, bufferLevel, horizonRateList)
	for bitrate in horizonRateList:
	
		download_time = downloadTimeCalculation(throughput,bitrate) * 1000. # ms
		
		rebufferDuration += max(int(download_time - bufferLevel), 0)#计算出rebuff duration

		bufferLevel = max(int(bufferLevel - download_time), 0)#计算当前的buffer occupancy
		
		bufferLevel = int(bufferLevel + segmentDuration * 1000.)
		

	
	videoQuality = sum(horizonRateList)
	
	horizonRateList.insert(0,preBitrate)
	
	qualityVar=sum([math.fabs(horizonRateList[i]-horizonRateList[i-1]) for i in range(1,len(horizonRateList))])

	utility = videoQuality - qualityVar - 25 * rebufferDuration
	# print(videoQuality, qualityVar, rebufferDuration)
	# breakpoint()

	
	return utility
	



def GetBest(throughput, preBitrate, bufferLevel):
	'''
	本方法旨在计算出给定throughput, preBitrate, bufferLevel下的最佳比特率
	:param throughput: 吞吐量
	:param preBitrate: 上一个segment的比特率
	:param bufferLevel: 缓冲区剩余量
	:return: bestRateList[UtilityList.index(max(UtilityList))]
			即最佳比特率
	'''

	UtilityList = []
	bestRateList = []
	
	#下面对选择了第一个比特率后产生的各种情况进行枚举，计算各个情况下的QoE

	for r0 in VrateList:#将要选择的第一个bitrate
		for r1 in VrateList:#将要选择的第二个bitrate
			for r2 in VrateList:#将要选择的第三个bitrate
				for r3 in VrateList:#将要选择的第四个bitrate
					for r4 in VrateList:#将要选择的第五个bitrate

						horizonRateList=[]
						horizonRateList.extend([r0])
						horizonRateList.extend([r1])
						horizonRateList.extend([r2])
						horizonRateList.extend([r3])
						horizonRateList.extend([r4])

						tmp = simulatePlay(throughput, preBitrate, bufferLevel, horizonRateList) #simulatePlay方法调用
						UtilityList.extend([tmp])				
						bestRateList.extend([r0])

	#选择QoE最大的比特率返回
	return bestRateList[UtilityList.index(max(UtilityList))]#比较QoE
	
def compute(throughput):
	'''
	本方法旨在计算出给定throughput,通过调用GetBest方法，枚举所有的吞吐量，buffer，bitrate确定的情况下的最优下一次的bitrate
	:param throughput:
	:return: 无,但是将信息以文件形式存储起来了
	'''
	print('computing...')



	f=open(str(throughput) + '.txt',"w") #generate file
	
	sentence=''
	for preBitrateIndex in range(len(VrateList)):#当前刚刚接收完的segment的bitrate
			
		for bufferLevelIndex in range(len(bufferLevelList)):#当前buffer level

			bestNextRate = GetBest(throughput, VrateList[preBitrateIndex], bufferLevelList[bufferLevelIndex])
	
			sentence+=(str(throughput) + '\t' + str(VrateList[preBitrateIndex]) + '\t' + str(bufferLevelList[bufferLevelIndex]) + '\t' + str(bestNextRate)+ '\t' +'\n')
			

	
	f.write(sentence)	
	
	f.close()
	
	# dispy_send_file(str(throughput) + '.txt') #transfer file (path)

for i in throughputList:
	print(f'{throughputList.index(i)} / {len(throughputList)}')
	compute(i)

#get files together
fr=open('MPC_balanced.txt','w')

for i in throughputList:
	f=open(str(i) + '.txt',"r")
	for lines in f:
		fr.write(lines)

	f.close()
	os.remove(str(i) + '.txt')
	
fr.close()


