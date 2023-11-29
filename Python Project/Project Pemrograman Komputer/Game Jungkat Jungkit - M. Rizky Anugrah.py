import pygame
import math as m

#button class
class Button:
    def __init__(self,text,width,height,pos,elevation,gui_font):
        #Core attributes 
        self.pressed = False
        self.elevation = elevation
        self.dynamic_elevation = elevation
        self.original_y_pos = pos[1]
        # top rectangle 
        self.top_rect = pygame.Rect(pos,(width,height))
        self.top_color = '#D74B4B'
        # bottom rectangle 
        self.bottom_rect = pygame.Rect(pos,(width,height))
        self.bottom_color = '#354B5E'
        #text
        self.text_surf = gui_font.render(text,True,'#FFFFFF')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)

    def draw(self,screen):
        # elevation logic 
        self.top_rect.y = self.original_y_pos - self.dynamic_elevation
        self.text_rect.center = self.top_rect.center 

        self.bottom_rect.midtop = self.top_rect.midtop
        self.bottom_rect.height = self.top_rect.height + self.dynamic_elevation

        pygame.draw.rect(screen,self.bottom_color, self.bottom_rect,border_radius = 12)
        pygame.draw.rect(screen,self.top_color, self.top_rect,border_radius = 12)
        screen.blit(self.text_surf, self.text_rect)
        self.check_click()

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            self.top_color = '#D74B4B'
            if pygame.mouse.get_pressed()[0]:
                self.dynamic_elevation = 0
                self.pressed = True
            else:
                if self.pressed == True:
                    self.pressed = False
        else:
            self.top_color = '#D74B4B'

    def reset(self):
        self.dynamic_elevation = self.elevation
        self.top_color = '#D74B4B'
        self.original_y_pos = self.original_y_pos
        
class Box():
    def __init__(self, image):
        self.image = image

    def draw(self,surface, x, y):
        surface.blit(self.image, (x, y-30))

def image_animasi(photo,width,height, scale ,frame, warna):
    image = pygame.Surface((width, height)).convert_alpha()
    image.blit(photo, (0,0), ((frame * width), 0, width, height))
    image = pygame.transform.scale(image, (width * scale, height * scale))
    image.set_colorkey(warna)

    return image

pygame.init()

#global ---------------------------------------------------------------------|
screen = (640, 640)
screen = pygame.display.set_mode(screen)
pygame.display.set_caption("Game Jungkat-Jungkit")
clock = pygame.time.Clock()
fps = 60

#properti lain ---------------------------------------------------------------------|
white = (255,255,255);black = (0,0,0);green = (52,235,161)
red = (255,0,0);cyan = (52,235,216); brown = (100,84,82)
dyel = (82,81,0)
font1 = pygame.font.SysFont('microsoftsansserif',20)
font2 = pygame.font.SysFont('microsoftsansserif',45)
font3 = pygame.font.SysFont('microsoftsansserif',25)
font4 = pygame.font.SysFont('microsoftsansserif',15)

#background ---------------------------------------------------------------------|
bg = pygame.image.load("background.png").convert_alpha()
list_animasi = []
step = 25
waktu_akhir = pygame.time.get_ticks()
jedawaktu = 80 #dalam ms
frame = 0

for i in range(step):
    list_animasi.append(image_animasi(bg, 450, 300, 1.5, i,black))
# background = pygame.transform.scale(bg,(920,420))

#Button ---------------------------------------------------------------------|
#bagian bawah
b_tambahkan = Button('Tambahkan',110,30,(30,185),5,font1)
b_reset = Button('Reset',60,30,(175,185),5,font1)
b_start = Button('Mulai',60,30,(295,185),5,font1)
#kiri
b_kiri = Button('Kiri',60,30,(55,70),5,font1)
b_kanan = Button('Kanan',60,30,(55,120),5,font1)
#tengah
b_5kg = Button('5 Kg',60,30,(175,70),5,font1)
b_10kg = Button('10 Kg',60,30,(175,120),5,font1)
#kiri
b_05m = Button('0,5 m',60,30,(295,70),5,font1)
b_1m = Button('1 m',60,30,(295,120),5,font1)

#properti ketika game mulai 
draw = 0
run = 0
kiri = 0
posisi = 0
berat = 0
l_box = [0,0,0,0]

#box ---------------------------------------------------------------------|
box_img = pygame.image.load("box1.png")
box_img = pygame.transform.scale(box_img,(30,30))
box_img2 = pygame.image.load("box2.png")
box_img2 = pygame.transform.scale(box_img2,(30,60))
box_x = 120
box_y = 525
box_y2 = 495

#properti jungkat-jungkit ---------------------------------------------------------------------|
stick_j = pygame.image.load("stick.png")
stick_j = pygame.transform.scale(stick_j, (400,10))
stick_x = 320
stick_y = 530
to_a = 0
to_b = 0
angle = 0
kom_x = 0
kom_y1 = 0
kom_y2 = 0
l_to = [0,0]

#tulisan ----------|
jdl_toa = font1.render("\u03A3\u03C4 Kiri",True,black)
jdl_tob = font1.render("\u03A3\u03C4 Kanan",True,black)
jdl_posisi = font1.render("Posisi",True,red)
jdl_berat = font1.render("Berat",True,red)
jdl_jarak = font1.render("Jarak",True,red)
warcop = font1.render("Challenge WARCOP 2022",True,red)
nama = font4.render("Muhammad Rizky Anugrah - UNJ",True,black)

#main ---------------------------------------------------------------------|
done = False
while done == False:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill(green)

    waktu_sekarang = pygame.time.get_ticks()
    if waktu_sekarang - waktu_akhir >= jedawaktu:
        frame += 1
        waktu_akhir = waktu_sekarang

        if frame >= len(list_animasi):
            frame = 0

    screen.blit(list_animasi[frame],[0,230])

    #judul tabel
    screen.blit(jdl_posisi,[55,25])
    screen.blit(jdl_berat,[175,25])
    screen.blit(jdl_jarak,[295,25])

    #Tabel
    pygame.draw.line(screen, black, (20,20),(380,20),4)
    pygame.draw.line(screen, black, (20,50),(380,50),4)
    pygame.draw.line(screen, black, (20,170),(380,170),4)
    pygame.draw.line(screen, black, (20,220),(20,20),4)
    pygame.draw.line(screen, black, (140,170),(140,20),4)
    pygame.draw.line(screen, black, (260,170),(260,20),4)
    pygame.draw.line(screen, black, (380,220),(380,20),4)
    pygame.draw.line(screen, black, (20,220),(380,220),4)

    #tulisan kanan
    screen.blit(warcop,[395,20])
    screen.blit(nama,[400,50])   

    #jungkat jungkit ---------------------------------------------------------------|
    stick_rotate = pygame.transform.rotate(stick_j,angle)
    st_w = stick_x - int(stick_rotate.get_width()/2)
    st_h = stick_y - int(stick_rotate.get_height()/2)
    screen.blit(stick_rotate,((st_w),(st_h)))
    segitiga = pygame.draw.polygon(screen, black, points=[(300,600), (320,530), (340,600)]) # point kiri, atas, kanan
    bulat = pygame.draw.circle(screen,red,[stick_x,stick_y],10)
    tulisan_perb = font1.render("...",True,black)

    if run == 1:
        #to = W * Lk, W = l_box * 5, l_box = 1 atau 2
        to_a = l_box[0]*5*1 + l_box[1]*5*0.5
        to_b = l_box[2]*5*1 + l_box[3]*5*0.5

        l_to = [to_a,to_b]
        
        if st_w == 130 and st_h == 461:
            to_a = 0
            to_b = 0
        
        if to_a > to_b:
            angle += 0.5
            kom_x -= 5/38
            kom_y1 -= 10/38
            kom_y2 += 5/38
        
        if to_a < to_b:
            angle -= 0.5
            kom_x += 5/38
            kom_y1 += 5/38
            kom_y2 -= 5/38

        if l_to[0] > l_to[1]:
            tulisan_perb = font1.render(">",True,black)

        if l_to[0] < l_to[1]:
            tulisan_perb = font1.render("<",True,black)

        if l_to[0] == l_to[1]:
            tulisan_perb = font1.render("=",True,black)

    #tampilan perhitungan ---------------------------------------------|
    tulisan_toa = font1.render(f'{str(l_to[0])}N',True,black)
    tulisan_tob = font1.render(f'{str(l_to[1])}N',True,black)
    screen.blit(tulisan_toa,[415,130])
    screen.blit(tulisan_perb,[485,130])
    screen.blit(tulisan_tob,[525,130])
    screen.blit(jdl_toa,[410,90])
    screen.blit(jdl_tob,[510,90])

    #box ---------------------------------------------------------------|
    porosx = 320
    porosy = 525
    porosy2 = 495
    box_rotate = pygame.transform.rotate(box_img,angle)
    box_rotate2 = pygame.transform.rotate(box_img2,angle)

    #rotasi ---------------------------------------------------------------|
    angle_con = m.radians(angle)
    pos_bx = 200*m.cos(angle_con)
    pos_by = 200*m.sin(angle_con)

    # drawbox ---------------------------------------------------------------|
    if l_box[0] == 1:
        boxL1 = Box(box_rotate) 
        boxL1.draw(screen, porosx - pos_bx + kom_x, porosy + pos_by + kom_y1)
    if l_box[1] == 1:
        boxL2 = Box(box_rotate)
        boxL2.draw(screen, porosx - pos_bx/2 + kom_x, porosy + pos_by/2 + kom_y1)
    if l_box[2] == 1:
        boxR1 = Box(box_rotate) 
        boxR1.draw(screen, porosx + pos_bx - 30 + kom_x, porosy - pos_by + kom_y2)
    if l_box[3] == 1:
        boxR2 = Box(box_rotate)
        boxR2.draw(screen, porosx + pos_bx/2 + kom_x, porosy - pos_by/2 - kom_y2)
    
    if l_box[0] == 2:
        boxL1 = Box(box_rotate2) 
        boxL1.draw(screen, porosx - pos_bx + kom_x*2, porosy2 + pos_by + kom_y1)
    if l_box[1] == 2:
        boxL2 = Box(box_rotate2)
        boxL2.draw(screen, porosx - pos_bx/2 + kom_x*2, porosy2 + pos_by/2 + kom_y1)
    if l_box[2] == 2:
        boxR1 = Box(box_rotate2) 
        boxR1.draw(screen, porosx + pos_bx - 30 + kom_x*2, porosy2 - pos_by + kom_y2)
    if l_box[3] == 2:
        boxR2 = Box(box_rotate2)
        boxR2.draw(screen, porosx + pos_bx/2 + kom_x*2, porosy2 - pos_by/2 - kom_y2)

    #kondisi memilih ---------------------------------------------------------|
    if draw == 1:
        if berat == 1: # berat = 1 (5kg), berat = 2 (10kg)
            if kiri == 1: #kiri = 1 (kiri), kiri = 2 (kanan)
                if posisi == 1: #posisi = 1 (1m), posisi = 2 (0,5m)
                    l_box[0] = 1
                elif posisi == 2:
                    l_box[1] = 1
            elif kiri == 2:
                if posisi == 1:
                    l_box[2] = 1
                elif posisi == 2:
                    l_box[3] = 1
        elif berat == 2:
            if kiri == 1:
                if posisi == 1:
                    l_box[0] = 2
                elif posisi == 2:
                    l_box[1] = 2
            elif kiri == 2:
                if posisi == 1:
                    l_box[2] = 2
                elif posisi == 2:
                    l_box[3] = 2
        draw = 0
    
    #button ---------------------------------------------------------------|
    b_tambahkan.draw(screen)
    b_reset.draw(screen)
    b_start.draw(screen)
    b_kiri.draw(screen)
    b_kanan.draw(screen)
    b_5kg.draw(screen)
    b_10kg.draw(screen)
    b_1m.draw(screen)
    b_05m.draw(screen)

    if b_tambahkan.pressed == True:
        # tombol naik turun
        b_tambahkan.dynamic_elevation = 0
        b_tambahkan.draw(screen)
        b_tambahkan.reset()
        draw = 1
    
    #pasangan reset-mulai
    if b_reset.pressed == True:
        b_reset.dynamic_elevation = 0
        b_reset.draw(screen)
        b_reset.reset()
        #reset all
        b_start.reset()
        b_tambahkan.reset()
        b_kanan.reset()
        b_kiri.reset()
        b_5kg.reset()
        b_10kg.reset()
        b_1m.reset()
        b_05m.reset()
        #properti reset
        draw = 0
        run = 0
        kiri = 0
        posisi = 0
        to_a = 0
        to_b = 0
        angle = 0
        st_w = 120
        st_h = 530
        kom_x = 0
        kom_y1 = 0
        kom_y2 = 0
        l_box = [0,0,0,0]
        l_to = [0,0]

    if b_start.pressed == True:
        b_start.dynamic_elevation = 0
        b_start.draw(screen)
        b_start.reset()
        b_reset.reset()
        run = 1

    #pasangan kiri-kanan
    if b_kiri.pressed == True:
        b_kanan.reset()
        kiri = 1

    if b_kanan.pressed == True:
        b_kiri.reset()
        kiri = 2

    #pasangan 5kg 10 kg
    if b_5kg.pressed == True:
        b_10kg.reset()
        berat = 1
        
    if b_10kg.pressed == True:
        b_5kg.reset()
        berat = 2

    #pasangan 0.5 - 1 m
    if b_05m.pressed == True:
        b_1m.reset()
        posisi = 2

    if b_1m.pressed == True:
        b_05m.reset()
        posisi = 1
        
    pygame.display.update()
    clock.tick(fps)

print("EXITED PROGRAM")
pygame.quit()