def Gambarlayar():
    layar.fill(green)
    layar.blit(text1,[300,10])
    pygame.draw.line(layar,darkgray,(0,600),(1000,600),20)

def GambarBola(x,y,warna):
    pygame.draw.circle(layar,warna,[x,y],25)

def Gerak_jatuh(xx,yy,z):
    while True:
        z = z + a
        yy = yy + z
        Gambarlayar()
        GambarBola(xx,yy,white)
        pygame.display.flip()
        clock.tick(ct)
        if yy >= 580:
            break
    return xx,yy,z

def Gerak_atas(xx,yy,z):
    while True:
        z = z - a
        yy = yy - z
        Gambarlayar()
        GambarBola(xx,yy,white)
        pygame.display.flip()
        clock.tick(ct)
        if z <= 0:
            break
    return xx,yy,z

def GerakBola(x,y,z):
    while True:
        if z <= 0:
            x,y,z = Gerak_jatuh(x,y,z)
        elif y >= 580:
            x,y,z = Gerak_atas(x,y,z)
            if y >=580:
                break
    return x,y,z

import pygame, sys
from pygame.locals import * 
pygame.init()

#Warna
green=(0,176,80)
white=(250,250,250)
darkgray=(50,50,50)

#Layar
layar = pygame.display.set_mode((1000, 600))
pygame.display.set_caption('Simulasi Bola Jatuh Bebas dan Memantul')
layar.fill(green)
pygame.draw.line(layar,darkgray,(0,600),(1000,600),20)

#Text
font1 = pygame.font.Font(None,30)
text1 = font1.render("Simulasi Bola Jatuh Bebas & Memantul",True,white)
layar.blit(text1,[300,10])

#Nilai Awal
x = 500; y = 50
z = 0; a = 3
ct = 40
clock = pygame.time.Clock()

#Program utama pygame
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            x,y,z = GerakBola(x,y,z)
    GambarBola(x,y,white)
    pygame.display.flip()
    clock.tick(ct)