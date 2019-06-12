# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:55:03 2017

@author: davidson
"""


#Definições
numNeuronio = 30
numEntrada = 1
numSaida = 1

import numpy as np

def sigmoid(soma):
    return 1/(1+np.exp(-soma))
    #return 2/(1+np.exp(-2*soma)) - 1        #tangente hiperbolica
    #return np.exp(-soma*soma)               #exponencial


def sigmoidDerivada(sig):
    return sig*(1-sig)
    #return 1-sig*sig                        #derivada da tangente hip
    #return -2*sig*np.exp(-sig*sig)          #derivada da exponencial


arq = open("plotB7.dat", "r")
a = []
b = []

for l in arq:
    v = l.split()
    v[0] = float(v[0])
    v[1] = float(v[1])
    b.append([v[1]])
    a.append([v[1]])

entradas = np.array(a)
#print(mat)

#entradas = np.array([[0,0],
#                     [0,1],
#                     [1,0],
#                     [1,1]])

#saidas = np.array([[0], [1], [1], [0]])
saidas = np.array(b)

#pesos de camada de entrada para primeira camada escondida
#pesos0 = np.array([[-0.424, -0.740, -0.961],
#                   [0.358, -0.577, -0.469]])
pesos0 = 2*np.random.random((numEntrada,numNeuronio))-1                  

#pesos da camada oculta para camada de saída
#pesos1 = np.array([[-0.017], [-0.893], [0.148]])
pesos1 = 2*np.random.random((numNeuronio,numSaida))-1

epocas = 10000
taxaAprendizagem = 0.01
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(abs(erroCamadaSaida))
    print("Erro: "+str(mediaAbsoluta))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    #derivadaSaida = sigmoidDerivada(somaSinapse1)
    deltaSaida = erroCamadaSaida*derivadaSaida
    
    pesos1Transposta = pesos1.T             #faz transposta
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso*sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1*momento)-(pesosNovo1*taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0*momento)-(pesosNovo0*taxaAprendizagem)
    
    
