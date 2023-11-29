import pygame
import pygame.freetype

#button class

class tombol:
    def __init__(self,tulisan,width,height,pos,perubahan,gui_font):
        self.tekan = False
        self.perubahan = perubahan
        self.animasi = perubahan
        self.posy_awal = pos[1]
        self.rect_atas = pygame.Rect(pos,(width,height))
        self.warna_atas = '#3E926A'
        self.rect_bawah = pygame.Rect(pos,(width,height))
        self.warna_bawah = '#000000'
        self.text_surf = gui_font.render(tulisan,True,'#FFFFFF')
        self.text_rect = self.text_surf.get_rect(center = self.rect_atas.center)
        list_tombol.append(self)

    def gambar(self,screen):
        self.rect_atas.y = self.posy_awal - self.animasi
        self.text_rect.center = self.rect_atas.center 
        self.rect_bawah.midtop = self.rect_atas.midtop
        self.rect_bawah.height = self.rect_atas.height + self.animasi
        pygame.draw.rect(screen,self.warna_bawah, self.rect_bawah)
        pygame.draw.rect(screen,self.warna_atas, self.rect_atas)
        screen.blit(self.text_surf, self.text_rect)
        self.klik_mouse()

    def klik_mouse(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect_atas.collidepoint(mouse_pos):
            self.warna_atas = '#3E926A'
            if pygame.mouse.get_pressed()[0]:
                self.animasi = 0
                self.tekan = True   
            else:
                self.animasi = self.perubahan
                if self.tekan == True:
                    self.tekan = False
        else:
            self.animasi = self.perubahan
            self.warna_atas = '#3E926A'

def image_animasi(photo,width,height, scale ,frame, warna):
    image = pygame.Surface((width, height)).convert_alpha()
    image.blit(photo, (0,0), ((frame * width), 0, width, height))
    image = pygame.transform.scale(image, (width * scale, height * scale))
    image.set_colorkey(warna)

    return image

def gambar_tombol():
    for i in list_tombol:
        i.gambar(screen)

pygame.init()

#global ---------------------------------------------------------------------|
screen = (640, 640)
screen = pygame.display.set_mode(screen)
pygame.display.set_caption("Countdown Timer")
clock = pygame.time.Clock()
fps = 60

#properti lain ---------------------------------------------------------------------|
white = (255,255,255);black = (0,0,0);green = (52,235,161)
red = (255,0,0);cyan = (52,235,216); brown = (100,84,82)
blue = (14,96,230)
font1 = pygame.font.SysFont('microsoftsansserif',20)
font2 = pygame.font.SysFont('microsoftsansserif',25)
font3 = pygame.font.SysFont('microsoftsansserif',40)
list_tombol = []

#background -----------------------------------------------------------------------|
bg = pygame.image.load("D:\\My Project\\Program Countdown Timer\\background.png").convert_alpha()
list_animasi = []
step = 25
waktu_akhir = pygame.time.get_ticks()
jedawaktu = 80 #dalam ms
frame = 0

for i in range(step):
    list_animasi.append(image_animasi(bg, 450, 300, 1.5, i,black))

#animasi jam
gambar_jam = pygame.image.load("D:\\My Project\\Program Countdown Timer\\jam.png").convert_alpha()
list_animasi2 = []
step2 = 4
waktu_akhir2 = pygame.time.get_ticks()
jedawaktu2 = 200
frame2 = 0

for i in range(step2):
    list_animasi2.append(image_animasi(gambar_jam, 500, 591, 0.2, i,black))

#kotak waktu
kotak_waktu = pygame.image.load("D:\\My Project\\Program Countdown Timer\\kotakwaktu.png").convert_alpha()
kotak_waktu.set_alpha(200)

#output timer -----------------------------------------------------------------------|
font_output = pygame.freetype.SysFont(None,50)
font_output.origin = True
waktu = 36000
konstan_waktu = 0 

#tulisan timer
t_jam = font1.render("Jam",True,white)
t_menit = font1.render("Menit",True,white)
t_detik = font1.render("Detik",True,white)
t_timer = font3.render("TIMER",True,black)

#tombol
b_tb_jam = tombol("+ Jam",80,40,(40,485),7,font2)
b_k_jam = tombol("- Jam",80,40,(40,565),7,font2)
b_tb_menit = tombol("+ Menit",100,40,(160,485),7,font2)
b_k_menit = tombol("- Menit",100,40,(160,565),7,font2)
b_tb_detik = tombol("+ Detik",90,40,(290,485),7,font2)
b_k_detik = tombol("- Detik",90,40,(290,565),7,font2)

b_mulai = tombol("Mulai",70,30,(530,480),5,font1)
b_pause = tombol("Pause",70,30,(530,530),5,font1)
b_reset = tombol("Reset",70,30,(530,580),5,font1)

#main ---------------------------------------------------------------------|
done = False
while done == False:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    
    screen.fill(blue)
    pygame.draw.line(screen, black, (0,450),(640,450),10)
    pygame.draw.line(screen, black, (3,450),(3,640),10)
    pygame.draw.line(screen, black, (635,450),(635,640),10)
    pygame.draw.line(screen, black, (0,635),(640,635),10)

    #animasi background
    waktu_sekarang = pygame.time.get_ticks()
    if waktu_sekarang - waktu_akhir >= jedawaktu:
        frame += 1
        waktu_akhir = waktu_sekarang

        if frame >= len(list_animasi):
            frame = 0

    screen.blit(list_animasi[frame],[0,0])

    #animasi jam
    waktu_sekarang2 = pygame.time.get_ticks()
    if waktu_sekarang2 - waktu_akhir2 >= jedawaktu2:
        frame2 += 1
        waktu_akhir2 = waktu_sekarang2

        if frame2 >= len(list_animasi2):
            frame2 = 0

    screen.blit(list_animasi2[frame2],[405,480])

    #perhitungan timer
    jam = int(waktu / 86400  % 24)
    menit = int(waktu / 3600 % 60)
    detik = int(waktu / 60  % 60)
    waktu -= konstan_waktu
            
    if waktu == 0:
        konstan_waktu = 0

    #Tampilan Timer
    output_jam = f"{jam:02d}"
    output_menit = f"{menit:02d}"
    output_detik = f"{detik:02d}"
    screen.blit(t_timer,[260,15])
    screen.blit(kotak_waktu, (110,170))
    screen.blit(kotak_waktu, (270,170))
    screen.blit(kotak_waktu, (430,170))
    font_output.render_to(screen,[140,230], output_jam, white)
    font_output.render_to(screen,[300,230], output_menit, white)
    font_output.render_to(screen,[460,230], output_detik, white)
    screen.blit(t_jam,[140,255])
    screen.blit(t_menit,[300,255])
    screen.blit(t_detik,[460,255])

    gambar_tombol()
    if b_tb_jam.tekan == True:
        waktu += 43200
    
    if b_k_jam.tekan == True:
        waktu -= 43200

    if b_tb_menit.tekan == True:
        waktu += 720

    if b_k_menit.tekan == True:
        waktu -= 720

    if b_tb_detik.tekan == True:
        waktu += 60

    if b_k_detik.tekan == True:
        waktu -= 60 
    
    if b_mulai.tekan == True:
        konstan_waktu = 1  

    if b_pause.tekan == True:
        konstan_waktu = 0  

    if b_reset.tekan == True:
        waktu = 0
        konstan_waktu = 0  

    pygame.display.update()
    clock.tick(fps)

print("EXITED PROGRAM")
pygame.quit()