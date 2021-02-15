# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:21:34 2020

@author: Harry Bowman
"""
##################################################################################
#Model
##################################################################################
class UNetModel1D(torch.nn.Module):
    def __init__(self, n_levels, conv_per_level, base_size, scale_size, skipConn, kernel_size=3):
        super(UNetModel1D, self).__init__()
        self.enc_layers = torch.nn.ModuleList()
        self.dec_layers = torch.nn.ModuleList()
        self.n_levels = n_levels
        self.conv_per_level = conv_per_level
        self.base_size = base_size
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.skipConn = skipConn
        self.inputs = 1
        self.outputs = 1
        self.mean = 0
        self.std = 0
        #print(scale_size)
        
        for i in range(self.n_levels):
            self.enc_layers.append(torch.nn.ModuleList())
            num_filts = int(self.base_size*(self.scale_size**(self.n_levels - i-1)))
            for j in range(self.conv_per_level):
                insize = self.inputs if i == 0 and j == 0 else num_filts if j!=0 else int(num_filts*(scale_size)) #inputs for very first layer are 1, and first conv in each level are the size of the prev.
                self.enc_layers[-1].append(torch.nn.Conv1d(insize,num_filts,self.kernel_size,1,padding=self.kernel_size//2))
        
        for i in range(self.n_levels-1,-1,-1):
            self.dec_layers.append(torch.nn.ModuleList())
            num_filts = int(self.base_size*(self.scale_size**(self.n_levels - i - 1)))
            if i!=self.n_levels-1:  #don't add to the bottom layer
                for j in range(self.conv_per_level):
                    outsize = self.outputs if i == 0 and j == self.conv_per_level-1 else num_filts
                    self.dec_layers[-1].append(torch.nn.Conv1d(num_filts,outsize,self.kernel_size,1,padding=self.kernel_size//2))
            if i!=0:
                self.dec_layers[-1].append(torch.nn.ConvTranspose1d(num_filts,int(num_filts*scale_size),self.kernel_size,stride=2,padding=self.kernel_size//2, output_padding=1))
    
    def activation_fn(self,x,norm=True):
        x = F.elu(x)
        if norm:
            return F.layer_norm(x,x.size()[1:])
        else:
            return x

    def forward(self, x):
        level_outputs = []
        for I,i in enumerate(self.enc_layers):
            for J,j in enumerate(i):        
                x = self.activation_fn(j(x))
            level_outputs.append(x)
            if I!=len(self.enc_layers)-1:
                x = torch.nn.functional.max_pool1d(x, kernel_size=3, stride=2,padding=1) #filter size = 3, stride = 2
        for I,i in enumerate(self.dec_layers):
            if self.skipConn:
                if I>0 and I<len(self.dec_layers)-1:
                    x += level_outputs[len(self.enc_layers)-1-I]
            for J,j in enumerate(i):
                if I==len(self.dec_layers)-1 and J==len(i)-1:
                    x = j(x)#torch.clamp(j(x),-3,3)
                else:
                    x = self.activation_fn(j(x))        
        return x
    
    
##################################################################################
#DATALOADER
##################################################################################
#create the dataloader class - takes in a tuple of the velocities and spectrograms and adds them to the dataloader
# SORT OUT WHAT WE WANT TO DO WITH VELOCITY DATA
class spectrogramDataset(object):
    def __init__(self, tupleData):
        deviceID = tupleData[1][0]
        specList = tupleData[1][1]
        velList = tupleData[1][2]
        
        self.specs = [] #spectrogram data
        self.device = []
#         self.velocity = [] #velocity data
        self.index = [] #index of the data in chronological order
        for j,npSpec in enumerate(specList): #For eacfile (spectrum)
            #Add index
            self.index.append(j)  
            self.device.append(deviceID)
            
            #Add Spectrogram
            npSpecInt = [int(i) for i in npSpec] #Recast to int from string
            tensorData  = torch.Tensor(npSpecInt).unsqueeze(0)#add channels NOTE if you want to apply transofmations do so here 
            self.specs.append(tensorData)
        
            #Add velocity
#             self.velocity.append(torch.Tensor(float(velList[j])))

    def __getitem__(self, idx):
        return self.specs[idx]
    
    def __len__(self):
        return len(self.specs) 

class spectrogramDataset2(object):
    def __init__(self, specList):

        
        self.specs = [] #spectrogram data
        self.device = []
        self.timestamp = []
        self.index = [] #index of the data in chronological order
        
        for j,npSpec in enumerate(specList): #For eacfile (spectrum)
            #Add index
            self.index.append(j)  
            self.device.append(npSpec[0])
            self.timestamp.append(npSpec[1])
            
            #Add Spectrogram
            npSpecInt = npSpec[2:]
            tensorData  = torch.Tensor(npSpecInt).unsqueeze(0)#add channels NOTE if you want to apply transofmations do so here 
            self.specs.append(tensorData)
        
            #Add velocity
#             self.velocity.append(torch.Tensor(float(velList[j])))

    def __getitem__(self, idx):
        return self.specs[idx]
    
    def __len__(self):
        return len(self.specs) 
###########################################################################################
#Fully connected 1D auto encoder
###########################################################################################
class AutoencoderFC(nn.Module):
    def __init__(self,inputSize = 1024,levels = 3, scale = 4):
        #super(AutoencoderFC,self).__init__()
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        N = inputSize
        L = levels
        
        for i in range(L):
            n = inputSize if i==0 else int(N/(scale**i))
            self.encoders.append(nn.Linear(in_features = n, out_features = int(N/(scale**(i+1)))))
        for i in range(L):
            n = inputSize if i==L-1 else int(N/(scale**(L-1-i)))
            self.decoders.append(nn.Linear(in_features = int(N/(scale**(L-i))), out_features = n))
            
    def forward(self,x):
        '''x = self.encoder(x)
        x = self.decoder(x)'''
        x_list = [x]
        for i in self.encoders:
            x_list.append(F.relu(i(x_list[-1])))
            inp = x_list[-1]
        for I,i in enumerate(self.decoders):
            #add a previous output if we're not at the bottom of the U-Net nor the top
            # inp = inp if I==0 or I==len(self.decoders)-1 else inp+x_list[len(self.encoders)-I]
            
            if I!=len(self.decoders)-1 :
                inp = F.relu(i(inp)) 
                # print('Hidden Layer')
            else:
                # inp = torch.clamp(i(inp),0,1)
                inp = F.relu(i(inp))
                # inp = torch.sigmoid(i(inp))
                # print('outputlayer')

        return inp


##################################################################################
#UDFs
##################################################################################

schemaGVSD = StructType([
    StructField("deviceID", StringType()),
    StructField("timestamp", StringType()),
    StructField("Velocity", StringType()),
    StructField("temperature", StringType()),
    StructField("spectrum", ArrayType(StringType())),
    StructField("sampling_freq", StringType()),
    StructField("waveform_type", StringType()),
    StructField("size", StringType()), 
    StructField("count",IntegerType())
    
])

@pandas_udf(schemaGVSD, functionType=PandasUDFType.GROUPED_MAP)
def getValidSpecData(df):
    timeStamp = str(pd.to_numeric(pd.to_datetime(df['timestamp']))/1e9)
    deviceID = df['deviceID'].iloc[0]
    velocity = df['Velocity'].iloc[0]
    temperature = df['temperature'].iloc[0]
    spectrum = df['spectrum'].iloc[0]
    fs = df['sampling_freq'].iloc[0]
    
    #Check all waveform types and sizes are the same
    sVal = df['size'].iloc[0]
    wfVal = df['waveform_type'].iloc[0]
    sError = 0
    wfError = 0
    for i in df['size'].values:
        if i != sVal:
            sError=1
    for i in df['waveform_type'].values:
        if i != wfError:
            wfError=1
    if sError:
        sVal = 'error'
    if wfError:
        wvVal = 'error'
    if sError or wfError:
        deviceId = 'ERRORERROR'
    returnDf = pd.DataFrame([[deviceID, timeStamp[0], velocity,temperature, spectrum, fs, wfVal, sVal, df.shape[0]]])  
    return returnDf


##################################################################################
#Pandas Functions
##################################################################################
def specGrouper(df):
    #Add spectrograms into one long list
    d = df['deviceID'].iloc[0]
    specList = [df['spectrum'].iloc[0]]
    for i in df['spectrum']:
        specList.append(i)  
    specList.pop(0)
        
    velList = []
    for i in df['Velocity']:
        velList.append(i)
        
    newDf = pd.DataFrame()
#     newDf['deviceID'] = [d]
    newDf['spectrums'] = [specList]
    newDf['velocities'] = [velList]

    return newDf
    