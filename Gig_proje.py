import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import os

def sort_linesby_height(lines):
    new_lines = np.array([])
    for line in lines:
        line_lenght = abs(line[0][0] - line[0][2])** 2 + abs(line[0][1] - line[0][3])**2
        line_lenght = sqrt(line_lenght)
        line_lenght = int(line_lenght)
        x1, y1, x2, y2 = line.reshape(4)
        line = np.array([x1, y1, x2, y2, line_lenght])
        if new_lines.size == 0:
            new_lines = np.array([line])
        else:
            new_lines = np.append(new_lines, [line], axis=0)
    
    sorted_indices_asc = np.argsort(new_lines[:, 4])

    sorted_indices_desc = sorted_indices_asc[::-1]

    sorted_lines = new_lines[sorted_indices_desc]
    

    return sorted_lines


def combine_lines(lines):

    while len(lines) < 6:
        new_lines = []
        for line in lines:
            new_lines.append(line)
        lines.extend(new_lines)
    if len(lines) == 6:
            return lines
    
    for i in range((len(lines)) // 2):
        j = i + 1
        min_control = 2
        min_distance = 2000
        min_indis = len(lines)

        if len(lines) == 6:
            return lines
        while j < len(lines):
            if len(lines) == 6:
                return lines
            distance_start_tostart = (lines[i][0] - lines[j][0]) ** 2 + (lines[i][1] - lines[j][1]) ** 2
            distance_start_tostart = sqrt(distance_start_tostart)
            
            distance_start_toend = (lines[i][0] - lines[j][2]) ** 2 + (lines[i][1] - lines[j][3]) ** 2
            distance_start_toend = sqrt(distance_start_toend)
            
            distance_end_tostart = (lines[i][2] - lines[j][0]) ** 2 + (lines[i][3] - lines[j][1]) ** 2
            distance_end_tostart = sqrt(distance_end_tostart)
            
            distance_end_toend = (lines[i][2] - lines[j][2]) ** 2 + (lines[i][3] - lines[j][3]) ** 2
            distance_end_toend = sqrt(distance_end_toend)
            
            if distance_start_tostart < min_distance or distance_start_toend < min_distance or distance_end_tostart < min_distance or distance_end_toend < min_distance:
                m = min(distance_start_tostart,distance_start_toend,distance_end_tostart,distance_end_toend)
                if distance_start_tostart <= m:
                    min_distance = distance_start_tostart
                    min_indis = j
                    min_control = 0
                elif distance_start_toend <= m:
                    min_distance = distance_start_toend
                    min_indis = j
                    min_control = 1
                elif distance_end_tostart <= m:
                    min_distance = distance_end_tostart
                    min_indis = j
                    min_control = 2
                elif distance_end_toend <= m:
                    min_distance = distance_end_toend
                    min_indis = j
                    min_control = 3
            j += 1
        min_distance = int(min_distance)
        if min_indis != len(lines):
            if min_control == 0 and min_distance < 100:
                lines[i][2] = lines[min_indis][2]
                lines[i][3] = lines[min_indis][3]
                lines = np.delete(lines, min_indis, axis=0)
            elif min_control == 1 and min_distance < 100:
                lines[i][2] = lines[min_indis][2]
                lines[i][3] = lines[min_indis][3]
                lines = np.delete(lines, min_indis, axis=0)
            elif min_control == 2 and min_distance < 100:
                lines[i][2] = lines[min_indis][2]
                lines[i][3] = lines[min_indis][3]
                lines = np.delete(lines, min_indis, axis=0)
            elif min_control == 3 and min_distance < 100:
                lines[i][2] = lines[min_indis][2]
                lines[i][3] = lines[min_indis][3]
                lines = np.delete(lines, min_indis, axis=0)
        if len(lines) == 6:
            return lines

    return lines

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # 5 e 5 lik karelik alanı ortalama piksel rengi ile doldurur
    canny = cv2.Canny(blur, 50, 200)
    canny2 = cv2.Canny(gray, 50, 200)
    cv2.imshow("image_without_gaussian_blur", canny2)
    return canny

def display_lines(image, lines, number):
    
    line_image = np.zeros_like(image)
    if lines is None:
        return line_image, number
    if len(lines) <= 0:
        return line_image, number
    lines = sort_linesby_height(lines)
    for line in lines:
        #print(line[0][1])
            if abs(line[1] - line[3]) < 30: # dusey uzunlugu esik degerin altındaysa listeden cikarilir
                index_to_remove = np.where(np.all(lines == line, axis=1))[0]
                lines = np.delete(lines, index_to_remove, axis=0)
                continue
    if len(lines) <= 0:
        return line_image, number
    while len(lines) < 6:
        for line in lines:
            print("line_cloned")# len(lines ) < 6 clonedl ines to make it 6
            lines = np.append(lines, [line], axis=0)
    while len(lines) > 6:
        lines = combine_lines(lines)
    print("saved_for_ai:",lines)
    number = save_for_ai(image, lines, number)
    if lines is not None:
        for line in lines:
            #print(line)
            x1, y1, x2, y2 , line_length= line.reshape(5)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image, number

def save_for_ai(image, lines, number):
    
    labels_filename = number
    frames_filename = number
    
    labels_full_filename = f"{labels_filename}.npy"
    frames_full_filename = f"{frames_filename}.png"

    labels_directory = "labels"
    frames_directory = "frames"

    if not os.path.exists(labels_directory):    
        os.makedirs(labels_directory)
    if not os.path.exists(frames_directory):    
        os.makedirs(frames_directory)
        
    labels_file_path = os.path.join(labels_directory, labels_full_filename)    
    frames_full_filename = os.path.join(frames_directory, frames_full_filename)
    
    np.save(labels_file_path, lines)
    
    cv2.imwrite(frames_full_filename, image)
    number +=1
    return number
        
    

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(150, height), (650, height), (350, 100)]
        ])
    mask = np.zeros_like(image)
    
    cv2.fillPoly(mask, polygons, 255)
    cv2.imshow('masked_places', mask) # maskeledigimiz bolge
    
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def save_combo_image(frame, c_number):
    result_filename = c_number
        
    result_full_filename = f"{result_filename}.png"

    result_directory = "results"

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
              
    result_full_filename = os.path.join(result_directory, result_full_filename)
        
    cv2.imwrite(result_full_filename, frame)
    c_number += 1
    return c_number

cap = cv2.VideoCapture("test3.mp4")
number = 0
error_number = 0
combo_number = 0
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    
    if lines is not None:
        line_image, number = display_lines(frame, lines, number) # lines -> averaged_lines
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow('Gray_image', canny_image)
        cv2.imshow('Masked_gray_image', cropped_image)
        cv2.imshow('lines', line_image)
        cv2.imshow('lane_detection', combo_image)
        combo_number = save_combo_image(combo_image, combo_number)
    else:
        cv2.imshow('frames_without_lanes', frame)
        error_filename = error_number
        
        error_full_filename = f"{error_filename}.png"

        error_directory = "errors"

        if not os.path.exists(error_directory):
            os.makedirs(error_directory)
              
        error_full_filename = os.path.join(error_directory, error_full_filename)
        
        cv2.imwrite(error_full_filename, frame)
        error_number += 1
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
plt.imshow(canny)
plt.show()
    
# image = cv2.imread('duzyol3.png')
# lane_image = np.copy(image)
# canny = canny(lane_image)
# cropped_image = region_of_interest(canny)
# cv2.imshow('result3', cropped_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
# line_image = display_lines(lane_image, lines)
# cv2.imshow('result2', line_image)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result', combo_image)
# cv2.waitKey(0)
# plt.imshow(canny)
# plt.show()


