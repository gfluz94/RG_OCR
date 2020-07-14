import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\7700532380\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

class ImageDewarper(object):

    def __init__(self, blur_ksize=5, threshold_value=195, dilation_ksize=5, output_size=600):
        self.__blur_ksize = blur_ksize
        self.__dilation_ksize = dilation_ksize
        self.__output_size = output_size
        self.__threshold_value = threshold_value

    @property
    def blur_ksize(self):
        return self.__blur_ksize

    @property
    def dilation_ksize(self):
        return self.__dilation_ksize

    @property
    def output_size(self):
        return self.__output_size

    @property
    def threshold_value(self):
        return self.__threshold_value

    def __preprocess_img(self, img):
        blur = cv2.GaussianBlur(img, (self.__blur_ksize, self.__blur_ksize), 0)
        thresh = cv2.adaptiveThreshold(blur, self.__threshold_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
        return thresh

    def __get_contour(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area = -np.inf
        for contour in contours:
            area = cv2.contourArea(contour)
            if area>max_area:
                max_area = area
                roi = contour
        return roi

    def __generate_mask(self, img, contour):
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        cv2.drawContours(mask, [contour], 0, 0, 2)
        mask = cv2.dilate(mask, (self.__dilation_ksize, self.__dilation_ksize), iterations=10)
        return mask

    def __find_corners(self, mask):
        corners = cv2.goodFeaturesToTrack(mask, 4, 0.01, 10)
        corners = np.int0(corners)
        return np.float32(corners.reshape(-1,2))

    def __find_new_corners(self, v, high_value=600):
        idx = np.zeros(v.shape)
        v_copy = v.copy()
        x = v_copy[:,0]
        x_sorted = np.sort(v_copy[:,0])
        y = v_copy[:,1]
        x_array = []
        for element in x:
            for i, sorted_element in enumerate(x_sorted):
                if element==sorted_element:
                    x_array.append(i)
        idx[:,0] = x_array
        idx[:,1] = y.argsort()
        return np.float32((idx>1)*high_value)

    def run(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        preproc_img = self.__preprocess_img(img)
        contour = self.__get_contour(preproc_img)
        mask = self.__generate_mask(img, contour)
        pts1 = self.__find_corners(mask)
        pts2 = self.__find_new_corners(pts1, high_value=self.__output_size)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (self.__output_size, self.__output_size))
        return dst


class RGReader(object):
    def __init__(self, dewarper):
        self.__dewarper = dewarper
        self.__output_size = self.__dewarper.output_size
        self.__tesseract = pytesseract
        self.FIELDS = ["RG", "DATA_EXPED", "NOME", "NOME_MAE", "NOME_PAI", 
                       "DATA_NASC", "CPF", "CIDADE_ORIGEM"]
        self.COORDS = self.__get_coords()
        self.UF_TO_STATE = {
                'AC': 'Acre',
                'AL': 'Alagoas',
                'AP': 'Amapá',
                'AM': 'Amazonas',
                'BA': 'Bahia',
                'CE': 'Ceará',
                'DF': 'Distrito Federal',
                'ES': 'Espírito Santo',
                'GO': 'Goiás',
                'MA': 'Maranhão',
                'MT': 'Mato Grosso',
                'MS': 'Mato Grosso do Sul',
                'MG': 'Minas Gerais',
                'PA': 'Pará',
                'PB': 'Paraíba',
                'PR': 'Paraná',
                'PE': 'Pernambuco',
                'PI': 'Piauí',
                'RJ': 'Rio de Janeiro',
                'RN': 'Rio Grande do Norte',
                'RS': 'Rio Grande do Sul',
                'RO': 'Rondônia',
                'RR': 'Roraima',
                'SC': 'Santa Catarina',
                'SP': 'São Paulo',
                'SE': 'Sergipe',
                'TO': 'Tocantins'
        }

    @property
    def output_size(self):
        return self.__output_size

    def __get_coords(self):
        nrg_xy = [(67, 35), (204, 85)]
        exped_xy = [(406, 35), (538, 85)]
        nome_xy = [(21, 102), (566, 139)]
        mae_xy = [(21, 188), (566, 228)]
        pai_xy = [(21, 231), (566, 268)]
        natal_xy = [(21, 309), (394, 352)]
        nasc_xy = [(406, 309), (538, 352)]
        cpf_xy = [(20, 463), (188, 502)]
        coords_default = [nrg_xy, exped_xy, nome_xy , mae_xy, pai_xy, nasc_xy , cpf_xy, natal_xy]
        coords = []
        for ((x1, y1), (x2, y2)) in coords_default:
            x1 //= int(600/self.__output_size)
            y1 //= int(600/self.__output_size)
            x2 //= int(600/self.__output_size)
            y2 //= int(600/self.__output_size)
            coords.append([(x1, y1), (x2, y2)])
        return coords

    def read_img(self, img_path):
        target_img = self.__dewarper.run(img_path)
        info_extracted = {}
        for field, ((x1, y1), (x2, y2)) in zip(self.FIELDS, self.COORDS):
            roi = target_img[y1:y2, x1:x2]
            info_extracted[field] = self.__tesseract.image_to_string(roi)
        info_extracted["UF_ORIGEM"] = info_extracted["CIDADE_ORIGEM"].split("-")[-1]
        info_extracted["UF_ORIGEM"] = self.UF_TO_STATE[info_extracted["UF_ORIGEM"]].upper()
        info_extracted["CIDADE_ORIGEM"] = info_extracted["CIDADE_ORIGEM"].split("-")[0]
        return info_extracted


if __name__ == "__main__":
    dewarper = ImageDewarper(blur_ksize=5, threshold_value=195, dilation_ksize=5, output_size=600)
    rg_reader = RGReader(dewarper)
    output_rg = rg_reader.read_img("rg.jpg")