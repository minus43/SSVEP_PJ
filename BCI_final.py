#!/usr/bin/env python
# coding: utf-8

# In[1]:


#따로 다운받아야 하는 라이브러리
#시간이 걸리므로 미리 anaconda prompt에서 다운받는 것을 권장
#conda install scipy
#pip install psychopy
#conda install -c conda-forge emoji


# In[1]:


#필요한 모듈
import numpy as np
import matplotlib.pyplot as plt
import time
import psychopy
from psychopy import visual, core, event
from scipy.signal import find_peaks, peak_prominences
from numpy import matlib
from psychopy import visual, core, event
from multiprocessing import Process
import time
import emoji


# In[2]:


#간략한 가이드
#1. 시간을 확인한다
#1_1.현재시간이 오전일 경우 Data_pre 클래스의 load_file_path의 file_path 부분의 '_오후' -> '_오전'
#1_2. 현재시간이 낮 12시인 경우, file_path 부분의 str(time_2.tm_hour -12) -> str(time_2.tm_hour)
#1_3. 현재시간의 분의 자리가 10 이하인 경우, file_path 부분의 str(time_2.tm_min) 앞의 '_' -> '_0'

#2.mave를 기록하기 시작
#2_1.mave의 기록시간의 초 단위가 20초 이내일때 시작하고 바로 프로그램을 시작

class Data_pre:
    #load_file_path는 파일 읽기 자동화, 조건이 있다면 레코딩이 시작된 시점 중 분단위가 일치해야 함
    def load_file_path(self):
        time_1 = time.time()
        time_2 = time.localtime(time_1)
        file_path ='C:/MAVE_RawData/2021-0' + str(time_2.tm_mon) +'-' + str(time_2.tm_mday) + '_오전 ' + str(time_2.tm_hour)+'_'+ str(time_2.tm_min)+'/Fp1_FFT.txt'
        return file_path
    
    #delete_pre는 선행된 데이터들을 무시하기 위함 np.genfromtxt는 str을 읽지 못하고 숫자만 읽기 때문에 time에 관한 열을 읽지 못함
    #따라서 선행된 행이 얼마인지를 기억하면 데이터를 다시 읽을때 이 행만큼 무시함
    def delete_pre(self,file_path):
        data = np.genfromtxt(file_path , encoding='cp949', skip_header= 0 , usecols= 2)
        pre_data_len = len(data)
        print(pre_data_len)
        time.sleep(5)
        print(pre_data_len)
        return pre_data_len
    
    #find_peak는 delete_pre가 실행되고 난 뒤 바로 자극을 준 시간 이후에 실행되어야 함 
    #delete_pre가 실행되고 자극을 주면 자극 주기 전까지의 데이터 무시 가능
    #자극 실행 후 find_peak 실행하면 자극 끝난 시점의 데이터를 로드, 이때 skip_header에 전에 있던 delete_pre의 값을 넣어주면
    #자극 동안만의 시간에 대한 peak 검출
    #이때 자극 == 쉬는 시간 + 자극 시간 이라 가정하여 그 반의 시간을 len(data)/2로 하여 그 이후의 시간의 peak를 peak라 인식할 것
    def find_peak(self,file_path,pre_data_len):
        fdata_5 = np.array([])
        total_f_5 = 0.0
        fdata_48 = np.array([])
        total_f_48 = 0.0
        fdata_46 = np.array([])
        total_f_46 = 0.0
        fdata_44 = np.array([])
        total_f_44 = 0.0
        fdata_42 = np.array([])
        total_f_42 = 0.0
        
        fdata_12 = np.array([])
        total_f_12 = 0.0
        fdata_118 = np.array([])
        total_f_118 = 0.0
        fdata_116 = np.array([])
        total_f_116 = 0.0
        fdata_114 = np.array([])
        total_f_114 = 0.0
        fdata_112 = np.array([])
        total_f_112 = 0.0
        
        count_5 = 0
        count_12 = 0

        
        data = np.genfromtxt(file_path , encoding='cp949', skip_header= pre_data_len , usecols= (27,26,25,24,22,62,61,60,59,58))
        print(len(data))
        print(data)
        half = int(len(data)/2)
        for i in range(0,len(data)):
            fdata_5=np.append(fdata_5, data[i][0])
            total_f_5 += data[i][0]
            
            fdata_48=np.append(fdata_48, data[i][1])
            total_f_48 += data[i][1]
            
            fdata_46=np.append(fdata_46, data[i][2])
            total_f_46 += data[i][2]
            
            fdata_44=np.append(fdata_44, data[i][3])
            total_f_44 += data[i][3]
            
            fdata_42=np.append(fdata_42, data[i][4])
            total_f_42 += data[i][4]
            
            fdata_12=np.append(fdata_12, data[i][5])
            total_f_12 += data[i][5]
            
            fdata_118=np.append(fdata_118, data[i][6])
            total_f_118 += data[i][6]
            
            fdata_116=np.append(fdata_116, data[i][7])
            total_f_116 += data[i][7]
            
            fdata_114=np.append(fdata_114, data[i][8])
            total_f_114 += data[i][8]
            
            fdata_112=np.append(fdata_112, data[i][9])
            total_f_112 += data[i][9]
            
            
        rdata_5 = fdata_5/total_f_5*100
        rdata_48 = fdata_48/total_f_48*100
        rdata_46 = fdata_46/total_f_46*100
        rdata_44 = fdata_44/total_f_44*100
        rdata_42 = fdata_42/total_f_42*100
       
        
        rdata_12 = fdata_12/total_f_12*100
        rdata_118 = fdata_118/total_f_118*100
        rdata_116 = fdata_116/total_f_116*100
        rdata_114 = fdata_114/total_f_114*100
        rdata_112 = fdata_112/total_f_112*100
        
        peaks_5, _ = find_peaks(rdata_5, height = 9)   #5hz
        peaks_48, _ = find_peaks(rdata_48, height = 9)   #4.8hz
        peaks_46, _ = find_peaks(rdata_46, height = 9)   #4.6hz
        peaks_44, _ = find_peaks(rdata_44, height = 9)   #4.4hz
        peaks_42, _ = find_peaks(rdata_42, height = 9)   #4.2hz
        
        peaks_12, _ = find_peaks(rdata_12, height = 13)   #12hz
        peaks_118, _ = find_peaks(rdata_118, height = 13)   #11.8hz
        peaks_116, _ = find_peaks(rdata_116, height = 13)   #11.6hz
        peaks_114, _ = find_peaks(rdata_114, height = 13)   #11.4hz
        peaks_112, _ = find_peaks(rdata_112, height = 13)   #11.2hz
        
        for peak in peaks_5:
            if((peak > half)):
                count_5 += 1
                print("peak in 5")
                break
        for peak in peaks_48:
            if((peak > half)):
                count_5 += 1
                print("peak in 4.8")
                break
        for peak in peaks_46:
            if((peak > half)):
                count_5 += 1
                print("peak in 4.6")
                break
        for peak in peaks_44:
            if((peak > half)):
                count_5 += 1
                print("peak in 4.4")
                break
        for peak in peaks_42:
            if((peak > half)):
                count_5 += 1
                print("peak in 4.2")
                break
      
      
        for peak in peaks_12:
            if((peak > half)):
                count_12 += 1
                print("peak in 12")
                break
        for peak in peaks_118:
            if((peak > half)):
                count_12 += 1
                print("peak in 11.8")
                break
        for peak in peaks_116:
            if((peak > half)):
                count_12 += 1
                print("peak in11.6")
                break
        for peak in peaks_114:
            if((peak > half)):
                count_12 += 1
                print("peak in 11.4")
                break
        for peak in peaks_112:
            if((peak > half)):
                count_12 += 1
                print("peak in 11.2")
                break
                
        if(count_5 > count_12):
            return 5
        if(count_12 > count_5):
            return 12
        if(count_5 == count_12):
            return -1
        
class SSVEP_stimuli(object):
    
    def __init__(self, mywin= visual.Window([1600, 800], fullscr=False, monitor='testMonitor',units='deg', waitBlanking = False), trialdur = 3.0, numtrials=4, waitdur=2):
        
        self.mywin = mywin
        
        # colour for psychopy
        self.white = [1, 1, 1]
        self.black = [-1, -1, -1]
        self.red = [1, -1, -1]
        
        # frequency = 10Hz -- pattern 1 -- position:top
        self.pattern1_f0 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0, pos=[0, 400], size = 200,
                        color=self.white, colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        # frequency = 10Hz -- pattern 2 -- position:top                  
        self.pattern2_f0 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0, pos=[0, 400], size = 200,
                        color=self.black, colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)
        # frequency = 12Hz -- pattern 1 -- position:right
        self.pattern1_f1 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0, pos=[500, 0], size = 450,
                        color=[1,1,1], colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        # frequency = 12Hz -- pattern 2 -- position:right
        self.pattern2_f1 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0, pos=[500, 0], size = 450,
                        color=[-1,-1,-1], colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)
        # frequency = 15Hz -- pattern 1 -- position:left
        self.pattern1_f2 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0, pos=[-500, 0], size = 450,
                        color=[1,1,1], colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        # frequency = 15Hz -- pattern 2 -- position:left
        self.pattern2_f2 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0, pos=[-500, 0], size = 450,
                        color=[-1,-1,-1], colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)
        # frequency = 30Hz -- pattern 1 -- position:bottom
        self.pattern1_f3 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0, pos=[0, -400], size = 200,
                        color=[1,1,1], colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        # frequency = 30Hz -- pattern 2 -- position:bottom
        self.pattern2_f3 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0, pos=[0, -400], size = 200,
                        color=[-1,-1,-1], colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)
            
        

        self.text000 = visual.TextStim(win=self.mywin,text='상태양호/상태훌륭\n/동의/배고픔', pos=[500, 0], units='pix',height=50)

        self.text001 = visual.TextStim(win=self.mywin,text='상태양호/상태훌륭\n/동의/배고픔', pos=[500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text002 = visual.TextStim(win=self.mywin,text='아픔/ 매우 아픔/\n 졸림/ 거절', pos=[-500, 0], units='pix', height=50)

        self.text003 = visual.TextStim(win=self.mywin,text='아픔/ 매우 아픔/\n 졸림/ 거절', pos=[-500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)



        self.text100 = visual.TextStim(win=self.mywin,text='상태양호/상태훌륭', pos=[500, 0], units='pix',height=50)

        self.text101 = visual.TextStim(win=self.mywin,text='상태양호/상태훌륭', pos=[500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text102 = visual.TextStim(win=self.mywin,text='동의/배고픔', pos=[-500, 0], units='pix', height=50)

        self.text103 = visual.TextStim(win=self.mywin,text='동의/배고픔', pos=[-500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text104 = visual.TextStim(win=self.mywin,text='아픔/ 매우 아픔', pos=[500, 0], units='pix',height=50)

        self.text105 = visual.TextStim(win=self.mywin,text='아픔/ 매우 아픔', pos=[500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text106 = visual.TextStim(win=self.mywin,text='졸림/ 거절', pos=[-500, 0], units='pix', height=50)

        self.text107 = visual.TextStim(win=self.mywin,text='졸림/ 거절', pos=[-500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)



        self.text200 = visual.TextStim(win=self.mywin,text='상태양호', pos=[500, 0], units='pix',height=50)

        self.text201 = visual.TextStim(win=self.mywin,text='상태양호', pos=[500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text202 = visual.TextStim(win=self.mywin,text='상태훌륭', pos=[-500, 0], units='pix', height=50)

        self.text203 = visual.TextStim(win=self.mywin,text='상태훌륭', pos=[-500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text204 = visual.TextStim(win=self.mywin,text='동의', pos=[500, 0], units='pix',height=50)

        self.text205 = visual.TextStim(win=self.mywin,text='동의', pos=[500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text206 = visual.TextStim(win=self.mywin,text='배고픔', pos=[-500, 0], units='pix', height=50)

        self.text207 = visual.TextStim(win=self.mywin,text='배고픔', pos=[-500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text208 = visual.TextStim(win=self.mywin,text='아픔', pos=[500, 0], units='pix',height=50)

        self.text209 = visual.TextStim(win=self.mywin,text='아픔', pos=[500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text210 = visual.TextStim(win=self.mywin,text='매우 아픔', pos=[-500, 0], units='pix', height=50)

        self.text211 = visual.TextStim(win=self.mywin,text='매우 아픔', pos=[-500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text212 = visual.TextStim(win=self.mywin,text='졸림', pos=[500, 0], units='pix',height=50)

        self.text213 = visual.TextStim(win=self.mywin,text='졸림', pos=[500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

        self.text214 = visual.TextStim(win=self.mywin,text='거절', pos=[-500, 0], units='pix', height=50)

        self.text215 = visual.TextStim(win=self.mywin,text='거절', pos=[-500, 0], color=[-1,-1,-1], colorSpace='rgb', units='pix', height=50)

















        self.fixPos = [self.pattern1_f0.pos, self.pattern1_f1.pos, self.pattern1_f2.pos, self.pattern1_f3.pos]

        self.fixation = visual.GratingStim(win=self.mywin, color = self.red , size = 10, sf=0, colorSpace='rgb', units='pix')

        # frame array for 10Hz
        self.frame_f0 = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1,1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]

        # frame array for 12Hz
        self.frame_f1 = [1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1]

        # frame array for 15Hz
        self.frame_f2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]

        # frame array for 30Hz
        self.frame_f3 = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

        # frame array for 5hz
        self.frame_f4 = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
        
        self.trialdur = trialdur
        self.numtrials = numtrials
        self.waitdur = waitdur
        self.whatnum = 0

        # randomise sequence
        self.nBox = 4
        self.capBox = int(self.numtrials/self.nBox)
        self.aBox = np.arange(self.nBox)
        self.unshuffled = np.matlib.repmat(self.aBox, self.capBox, 1)
        self.randperm = np.random.permutation(self.numtrials)
        self.Boxes = self.unshuffled.ravel()
        self.Boxes = self.Boxes[self.randperm]
        print (self.Boxes)




        

    def stop(self):
        self.mywin.close()
        if self.whatnum==1:
            f = open('unicode.txt', 'w',encoding='UTF-8')
            f.write( emoji.emojize(':grinning_face:', use_aliases=True))
        elif self.whatnum==2:
            f = open('unicode.txt', 'w',encoding='UTF-8')
            f.write( emoji.emojize(':smiling_face_with_heart-eyes:', use_aliases=True))
        elif self.whatnum==3:
            f = open('unicode.txt', 'w',encoding='UTF-8')
            f.write( emoji.emojize(':OK_hand:', use_aliases=True))
        elif self.whatnum==4:
            f = open('unicode.txt', 'w',encoding='UTF-8')
            f.write( emoji.emojize(':fork_and_knife_with_plate:', use_aliases=True))
        elif self.whatnum==5:
            f = open('unicode.txt', 'w',encoding='UTF-8')
            f.write( emoji.emojize(':face_with_thermometer:', use_aliases=True))
        elif self.whatnum==6:
            f = open('unicode.txt', 'w',encoding='UTF-8')
            f.write( emoji.emojize(':knocked-out_face:', use_aliases=True))
        elif self.whatnum==7:
            f = open('unicode.txt', 'w',encoding='UTF-8')
            f.write( emoji.emojize(':sleepy_face:', use_aliases=True))
        elif self.whatnum==8:
            f = open('unicode.txt', 'w',encoding='UTF-8')
            f.write( emoji.emojize(':cross_mark:', use_aliases=True))
        f.close()
        core.quit()
        
          


    def start(self):
        
        # Loop through all trials
        self.fixation.setAutoDraw(True)
        self.Trialclock = core.Clock()
        # start_sample = self.client.last_sample
        # Loop through the required trial duration
        while self.Trialclock.getTime() < self.trialdur:
            #draws square and fixation on screen.             
            for frameN in range(len(self.frame_f0)):
                self.text000.setAutoDraw(True)
                self.text002.setAutoDraw(True)
                if self.frame_f4[frameN] == 1 :         
                    self.pattern1_f1.draw()
                if self.frame_f4[frameN] == -1 :
                    self.pattern2_f1.draw()
                if self.frame_f1[frameN] == 1 :
                    self.pattern1_f2.draw()
                if self.frame_f1[frameN] == -1 :
                    self.pattern2_f2.draw()               
                self.mywin.flip()
               
        #clean black screen off
        self.mywin.flip()
        self.fixation.setAutoDraw(False)
        #wait certain time for next trial
        core.wait(self.waitdur)
        #reset clock for next trial
        self.Trialclock.reset()    
        #count number of trials
        #print("Trial %d Complete" % self.count)

    def phase10(self):
        # Loop through all trials
        self.fixation.setAutoDraw(True)
        self.Trialclock = core.Clock()
        # start_sample = self.client.last_sample
        # Loop through the required trial duration
        while self.Trialclock.getTime() < self.trialdur:
            #draws square and fixation on screen.             
            for frameN in range(len(self.frame_f0)):
                self.text100.setAutoDraw(True)
                self.text102.setAutoDraw(True)
                if self.frame_f4[frameN] == 1 :         
                    self.pattern1_f1.draw()
                if self.frame_f4[frameN] == -1 :
                    self.pattern2_f1.draw()
                if self.frame_f1[frameN] == 1 :
                    self.pattern1_f2.draw()
                if self.frame_f1[frameN] == -1 :
                    self.pattern2_f2.draw()               
                self.mywin.flip()
               
        #clean black screen off
        self.mywin.flip()
        self.fixation.setAutoDraw(False)
        #wait certain time for next trial
        core.wait(self.waitdur)
        #reset clock for next trial
        self.Trialclock.reset()    
        #count number of trials
        #print("Trial %d Complete" % self.count)

    def phase11(self):
        # Loop through all trials
        self.fixation.setAutoDraw(True)
        self.Trialclock = core.Clock()
        # start_sample = self.client.last_sample
        # Loop through the required trial duration
        while self.Trialclock.getTime() < self.trialdur:
            #draws square and fixation on screen.             
            for frameN in range(len(self.frame_f0)):
                self.text104.setAutoDraw(True)
                self.text106.setAutoDraw(True)
                if self.frame_f4[frameN] == 1 :         
                    self.pattern1_f1.draw()
                if self.frame_f4[frameN] == -1 :
                    self.pattern2_f1.draw()
                if self.frame_f1[frameN] == 1 :
                    self.pattern1_f2.draw()
                if self.frame_f1[frameN] == -1 :
                    self.pattern2_f2.draw()               
                self.mywin.flip()
               
        #clean black screen off
        self.mywin.flip()
        self.fixation.setAutoDraw(False)
        #wait certain time for next trial
        core.wait(self.waitdur)
        #reset clock for next trial
        self.Trialclock.reset()    
        #count number of trials
        #print("Trial %d Complete" % self.count)

    def phase20(self):
        # Loop through all trials
        self.fixation.setAutoDraw(True)
        self.Trialclock = core.Clock()
        # start_sample = self.client.last_sample
        # Loop through the required trial duration
        while self.Trialclock.getTime() < self.trialdur:
            #draws square and fixation on screen.             
            for frameN in range(len(self.frame_f0)):
                self.text200.setAutoDraw(True)
                self.text202.setAutoDraw(True)
                if self.frame_f4[frameN] == 1 :         
                    self.pattern1_f1.draw()
                if self.frame_f4[frameN] == -1 :
                    self.pattern2_f1.draw()
                if self.frame_f1[frameN] == 1 :
                    self.pattern1_f2.draw()
                if self.frame_f1[frameN] == -1 :
                    self.pattern2_f2.draw()               
                self.mywin.flip()
               
        #clean black screen off
        self.mywin.flip()
        self.fixation.setAutoDraw(False)
        #wait certain time for next trial
        core.wait(self.waitdur)
        #reset clock for next trial
        self.Trialclock.reset()    
        #count number of trials
        #print("Trial %d Complete" % self.count)

    def phase21(self):
        # Loop through all trials
        self.fixation.setAutoDraw(True)
        self.Trialclock = core.Clock()
        # start_sample = self.client.last_sample
        # Loop through the required trial duration
        while self.Trialclock.getTime() < self.trialdur:
            #draws square and fixation on screen.             
            for frameN in range(len(self.frame_f0)):
                self.text204.setAutoDraw(True)
                self.text206.setAutoDraw(True)
                if self.frame_f4[frameN] == 1 :         
                    self.pattern1_f1.draw()
                if self.frame_f4[frameN] == -1 :
                    self.pattern2_f1.draw()
                if self.frame_f1[frameN] == 1 :
                    self.pattern1_f2.draw()
                if self.frame_f1[frameN] == -1 :
                    self.pattern2_f2.draw()               
                self.mywin.flip()
               
        #clean black screen off
        self.mywin.flip()
        self.fixation.setAutoDraw(False)
        #wait certain time for next trial
        core.wait(self.waitdur)
        #reset clock for next trial
        self.Trialclock.reset()    
        #count number of trials
        #print("Trial %d Complete" % self.count)

    def phase22(self):
        # Loop through all trials
        self.fixation.setAutoDraw(True)
        self.Trialclock = core.Clock()
        # start_sample = self.client.last_sample
        # Loop through the required trial duration
        while self.Trialclock.getTime() < self.trialdur:
            #draws square and fixation on screen.             
            for frameN in range(len(self.frame_f0)):
                self.text208.setAutoDraw(True)
                self.text210.setAutoDraw(True)
                if self.frame_f4[frameN] == 1 :         
                    self.pattern1_f1.draw()
                if self.frame_f4[frameN] == -1 :
                    self.pattern2_f1.draw()
                if self.frame_f1[frameN] == 1 :
                    self.pattern1_f2.draw()
                if self.frame_f1[frameN] == -1 :
                    self.pattern2_f2.draw()               
                self.mywin.flip()
               
        #clean black screen off
        self.mywin.flip()
        self.fixation.setAutoDraw(False)
        #wait certain time for next trial
        core.wait(self.waitdur)
        #reset clock for next trial
        self.Trialclock.reset()    
        #count number of trials
        #print("Trial %d Complete" % self.count)

    def phase23(self):
        # Loop through all trials
        self.fixation.setAutoDraw(True)
        self.Trialclock = core.Clock()
        # start_sample = self.client.last_sample
        # Loop through the required trial duration
        while self.Trialclock.getTime() < self.trialdur:
            #draws square and fixation on screen.             
            for frameN in range(len(self.frame_f0)):
                self.text212.setAutoDraw(True)
                self.text214.setAutoDraw(True)
                if self.frame_f4[frameN] == 1 :         
                    self.pattern1_f1.draw()
                if self.frame_f4[frameN] == -1 :
                    self.pattern2_f1.draw()
                if self.frame_f1[frameN] == 1 :
                    self.pattern1_f2.draw()
                if self.frame_f1[frameN] == -1 :
                    self.pattern2_f2.draw()               
                self.mywin.flip()
               
        #clean black screen off
        self.mywin.flip()
        self.fixation.setAutoDraw(False)
        #wait certain time for next trial
        core.wait(self.waitdur)
        #reset clock for next trial
        self.Trialclock.reset()    
        #count number of trials
        #print("Trial %d Complete" % self.count)

    def CTF(self):
        if (cnt_05==1 and cnt_15==1 and cnt_25==1):
            self.whatnum=8
        if (cnt_05==1 and cnt_15==1 and cnt_212==1):
            self.whatnum=7
        if (cnt_05==1 and cnt_112==1 and cnt_25==1):
            self.whatnum=6
        if (cnt_05==1 and cnt_112==1 and cnt_212==1):
            self.whatnum=5
        if (cnt_012==1 and cnt_15==1 and cnt_25==1):
            self.whatnum=4
        if (cnt_012==1 and cnt_15==1 and cnt_212==1):
            self.whatnum=3
        if (cnt_012==1 and cnt_112==1 and cnt_25==1):
            self.whatnum=2
        if (cnt_012==1 and cnt_112==1 and cnt_212==1):
            self.whatnum=1

if __name__ == "__main__":
    cnt=0
    cnt_012=0
    cnt_05=0
    cnt_15=0
    cnt_112=0
    cnt_25=0
    cnt_212=0
    stimuli = SSVEP_stimuli()
    user1 = Data_pre()
    path = user1.load_file_path()
    while cnt==0:
        delete_len = user1.delete_pre(path)
        stimuli.start()
        x=user1.find_peak(path,delete_len)
        if (x==5 or x==12):
            cnt=cnt+1
    while cnt==1:
        delete_len = user1.delete_pre(path)
        if x==5:
            cnt_05=1
            stimuli.phase11()
            x=user1.find_peak(path,delete_len)
        elif x==12:
            cnt_012=1
            stimuli.phase10()
            x=user1.find_peak(path,delete_len)

        if x==5:
            cnt_15=1
            cnt=cnt+1
        elif x==12:
            cnt_112=1
            cnt=cnt+1
    while cnt==2:
        delete_len = user1.delete_pre(path)
        if (cnt_05==1 and cnt_15==1)  :
            stimuli.phase23()
            x=user1.find_peak(path,delete_len)
        elif (cnt_05==1 and cnt_112==1):
            stimuli.phase22()
            x=user1.find_peak(path,delete_len)

        elif (cnt_012==1 and cnt_15==1) :
            stimuli.phase21()
            x=user1.find_peak(path,delete_len)

        elif (cnt_012==1 and cnt_112==1) :
            stimuli.phase20()
            x=user1.find_peak(path,delete_len)


        if x==5:
            cnt_25=1
            cnt=cnt+1
        elif x==12:
            cnt_212=1
            cnt=cnt+1
    stimuli.CTF()
    stimuli.stop()


# In[8]:





# In[ ]:




