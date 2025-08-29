import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import pygame
import matplotlib.pyplot as plt
from pygame.locals import *
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import deque
from dataclasses import dataclass

# -------------------- General Setup --------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter TF logs

# -------------------- Data --------------------
def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train, x_test, y_test)

def create_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

# -------------------- Model --------------------
def create_advanced_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    augmented = create_data_augmentation()(inputs)

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(augmented)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    y = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling2D(2)(y)
    y = tf.keras.layers.Dropout(0.25)(y)

    z = tf.keras.layers.Flatten()(y)
    z = tf.keras.layers.Dense(256, activation='relu')(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Dropout(0.5)(z)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(z)

    return tf.keras.Model(inputs, outputs)

def train_model(x_train, y_train, x_test, y_test):
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = create_advanced_model()

    def lr_scheduler(epoch, lr):
        if epoch > 10:
            return lr * tf.math.exp(-0.1)
        return lr

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]

    print("Training advanced model with data augmentation...")
    history = model.fit(x_train, y_train, epochs=50, batch_size=128,
                        validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)
    plot_training_history(history)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    return model, history

def plot_training_history(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout(); plt.savefig('training_history.png'); plt.close()

# -------------------- Explainability --------------------
def find_last_conv_layer(model):
    # robustly find last Conv2D layer for Grad-CAM
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # fallback to any layer by name if conv not found
    return model.layers[-2].name

def enhanced_predict(model, img, return_all=False):
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    img = img.astype('float32') / 255.0

    # center by moments
    try:
        m = cv2.moments(img)
        if m['m00'] != 0:
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])
        else:
            cx, cy = 14, 14
        M = np.float32([[1,0,14-cx],[0,1,14-cy]])
        img = cv2.warpAffine(img, M, (28,28))
    except:
        pass

    img_expanded = np.expand_dims(img, (0,-1))
    preds = model.predict(img_expanded, verbose=0)[0]
    idx = np.argmax(preds); conf = preds[idx]
    if return_all:
        return str(idx), conf, preds, img
    return str(idx), conf

def generate_grad_cam(model, img, layer_name=None):
    if layer_name is None:
        layer_name = find_last_conv_layer(model)
    img_expanded = np.expand_dims(img, (0, -1))

    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_expanded)
        top_index = tf.argmax(predictions[0])
        loss = predictions[:, top_index]

    grads = tape.gradient(loss, conv_outputs)[0]  # H,W,C
    weights = tf.reduce_mean(grads, axis=(0,1))
    conv_outputs = conv_outputs[0]

    cam = tf.zeros(conv_outputs.shape[:2], dtype=tf.float32)
    for i in range(conv_outputs.shape[-1]):
        cam += weights[i] * conv_outputs[:,:,i]
    cam = tf.nn.relu(cam)
    cam = cam.numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (28,28))
    return cam

# -------------------- Evaluation --------------------
def evaluate_model(model, x_test, y_test):
    print("Evaluating model on test set...")
    y_pred = model.predict(x_test, verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)
    acc = np.mean(y_pred_cls == y_test)
    print(f"Test accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred_cls)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Pred')
    plt.tight_layout(); plt.savefig('confusion_matrix.png'); plt.close()

    # Extra: classification report saved to file
    report = classification_report(y_test, y_pred_cls, digits=4)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("Saved confusion_matrix.png and classification_report.txt")

    mis_idx = np.where(y_pred_cls != y_test)[0]
    if len(mis_idx) > 0:
        plt.figure(figsize=(12,6))
        for i, idx in enumerate(mis_idx[:10]):
            plt.subplot(2,5,i+1)
            plt.imshow(x_test[idx].squeeze(), cmap='gray')
            plt.title(f'T:{y_test[idx]} P:{y_pred_cls[idx]}')
            plt.axis('off')
        plt.tight_layout(); plt.savefig('misclassified_examples.png'); plt.close()
        print(f"Saved {min(10,len(mis_idx))} misclassified_examples.png")

# -------------------- Settings --------------------
@dataclass
class AppSettings:
    camera_index: int = 0
    cam_width: int = 1280
    cam_height: int = 720
    roi_box: int = 240
    theme_dark: bool = True

SETTINGS = AppSettings()

# -------------------- Pygame UI Helpers --------------------
def init_pygame():
    pygame.init()
    try:
        pygame.mixer.init()
    except:
        pass

def draw_text(surface, text, pos, font, color):
    surface.blit(font.render(text, True, color), pos)

class Button:
    def __init__(self, rect, text, font, bg, bg_hover, fg, radius=10):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.bg = bg; self.bg_hover = bg_hover; self.fg = fg
        self.hover = False
    def draw(self, surface):
        color = self.bg_hover if self.hover else self.bg
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        pygame.draw.rect(surface, (30,30,30), self.rect, 2, border_radius=12)
        txt = self.font.render(self.text, True, self.fg)
        surface.blit(txt, txt.get_rect(center=self.rect.center))
    def update_hover(self, mouse):
        self.hover = self.rect.collidepoint(mouse)
    def clicked(self, event):
        return event.type == MOUSEBUTTONDOWN and event.button == 1 and self.hover

class Toggle:
    def __init__(self, rect, label, font, initial=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font
        self.value = initial
    def draw(self, surface):
        r = self.rect
        # switch
        pygame.draw.rect(surface, (80,80,80), r, border_radius=20)
        knob_x = r.x + (r.w-24 if self.value else 4)
        pygame.draw.circle(surface, (255,255,255), (knob_x+12, r.y + r.h//2), 12)
        # label
        surface.blit(self.font.render(self.label, True, (255,255,255)), (r.right+10, r.y-2))
    def toggle_on_click(self, event, mouse):
        if event.type == MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(mouse):
            self.value = not self.value
            return True
        return False

# -------------------- Camera Utilities --------------------
def preprocess_roi(gray, method="adaptive", blur=5, thresh_val=120):
    if blur > 0:
        k = blur if blur % 2 == 1 else blur+1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if method == "adaptive":
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    elif method == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Morphology to connect strokes
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return th

def auto_center_and_resize(th):
    # find largest contour to auto-crop
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        digit = th[max(y-10,0):y+h+10, max(x-10,0):x+w+10]
        if digit.size > 0:
            digit = cv2.resize(digit, (28,28))
            return digit
    return cv2.resize(th, (28,28))

# -------------------- Webcam Mode (Advanced) --------------------
def webcam_mode(model):
    cap = cv2.VideoCapture(SETTINGS.camera_index, cv2.CAP_DSHOW if os.name=='nt' else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SETTINGS.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SETTINGS.cam_height)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window = 'Digit Recognizer - Webcam'
    cv2.namedWindow(window)
    cv2.resizeWindow(window, 1280, 800)

    # Trackbars
    cv2.createTrackbar('Threshold', window, 120, 255, lambda x: None)
    cv2.createTrackbar('Blur', window, 5, 21, lambda x: None)
    cv2.createTrackbar('Min Confidence', window, 70, 100, lambda x: None)
    cv2.createTrackbar('ROI Size', window, SETTINGS.roi_box, 400, lambda x: None)

    mode_method = "adaptive"  # 'adaptive' | 'otsu' | 'manual'
    inference = False
    last_pred, last_conf = "None", 0.0
    heatmap = None
    ema_fps = None
    preds_cache = None

    recording = False
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print("Webcam Mode: SPACE=toggle recognize | A=Adaptive | O=Otsu | M=Manual | S=Snapshot | R=Record | E=Explain | Q=Quit")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        disp = frame.copy()
        h,w = disp.shape[:2]

        # ROI box
        box = cv2.getTrackbarPos('ROI Size', window)
        box = max(140, box)
        cx, cy = w//2, h//2
        x1,y1 = cx - box//2, cy - box//2
        x2,y2 = x1 + box, y1 + box
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 3)

        # FPS (EMA)
        t0 = time.time()

        # Controls
        threshold_val = cv2.getTrackbarPos('Threshold', window)
        blur_val = cv2.getTrackbarPos('Blur', window)
        blur_val = blur_val if blur_val % 2 == 1 else blur_val+1
        min_conf = cv2.getTrackbarPos('Min Confidence', window)/100.0

        if inference:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[y1:y2, x1:x2]
            th = preprocess_roi(roi, method=mode_method, blur=blur_val, thresh_val=threshold_val)
            auto = auto_center_and_resize(th)
            pred, conf, dist, proc = enhanced_predict(model, auto, return_all=True)
            last_pred, last_conf, preds_cache = pred, conf, dist

            if conf > min_conf:
                heatmap = generate_grad_cam(model, proc)
            # overlays
            small = cv2.cvtColor(cv2.resize(th, (150,150)), cv2.COLOR_GRAY2BGR)
            disp[10:160, w-160:w-10] = small

            # prob bars
            if w > 900:
                for i, p in enumerate(dist):
                    color = (0,255,0) if i == int(pred) else (170,170,170)
                    bar = int(p*150)
                    cv2.rectangle(disp, (w-200, 200+i*25), (w-200+bar, 200+i*25+18), color, -1)
                    cv2.putText(disp, f"{i}:{p*100:.1f}%", (w-220, 214+i*25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1)

            # heatmap preview
            if heatmap is not None and conf > min_conf:
                hm = np.uint8(255*cv2.resize(heatmap, (150,150)))
                hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                disp[170:320, w-160:w-10] = hm

            cv2.putText(disp, f"Prediction: {pred}", (10, h-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0) if conf>min_conf else (0,0,255), 2)
            cv2.putText(disp, f"Confidence: {conf*100:.1f}%", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if conf>min_conf else (0,0,255), 2)
            cv2.putText(disp, "RECOGNITION ACTIVE", (w-280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(disp, "RECOGNITION INACTIVE", (w-300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # FPS
        dt = max(1e-6, time.time()-t0)
        fps = 1.0/dt
        ema_fps = fps if ema_fps is None else 0.9*ema_fps + 0.1*fps
        cv2.putText(disp, f"FPS: {ema_fps:.1f}", (w-120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Recording
        if recording:
            if video_writer is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                video_writer = cv2.VideoWriter(f"session_{ts}.mp4", fourcc, 20.0, (w,h))
            video_writer.write(disp)
            cv2.putText(disp, "REC", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(disp, "SPACE: Toggle  |  A:Adaptive  O:Otsu  M:Manual  S:Snapshot  R:Record  E:Explain  Q:Quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(window, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            inference = not inference
        elif key == ord('a'):
            mode_method = "adaptive"
        elif key == ord('o'):
            mode_method = "otsu"
        elif key == ord('m'):
            mode_method = "manual"
        elif key == ord('s'):
            # snapshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"snapshot_{ts}.png", frame)
            print(f"Saved snapshot_{ts}.png")
        elif key == ord('r'):
            recording = not recording
            if not recording and video_writer is not None:
                video_writer.release(); video_writer = None
                print("Recording saved.")
        elif key == ord('e') and inference and heatmap is not None and last_conf > min_conf:
            # explanation window
            explanation = np.zeros((320, 640, 3), dtype=np.uint8)
            cv2.putText(explanation, f"Prediction: {last_pred}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(explanation, f"Confidence: {last_conf*100:.1f}%", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(explanation, "Heatmap (model attention) & Processed 28x28", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            hm = np.uint8(255*cv2.resize(heatmap, (150,150)))
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            explanation[140:290, 40:190] = hm

            # processed 28x28 preview
            if preds_cache is not None:
                # re-generate processed image for display (from last step)
                # (we don't store processed_img; recreate approximate)
                # This small re-run is cheap.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = gray[y1:y2, x1:x2]
                th = preprocess_roi(roi, method=mode_method, blur=blur_val, thresh_val=threshold_val)
                auto = auto_center_and_resize(th)
                auto_show = cv2.cvtColor(cv2.resize(auto, (150,150)), cv2.COLOR_GRAY2BGR)
                explanation[140:290, 230:380] = auto_show

            cv2.imshow('Prediction Explanation', explanation)
            cv2.waitKey(1)
        elif key == ord('q'):
            break

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

# -------------------- Drawing Pad (Advanced) --------------------
def drawing_pad_mode(model):
    init_pygame()
    WIDTH, HEIGHT = 1100, 760
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Digit Recognizer - Drawing Pad Mode')

    # Theme
    BG = (25,25,30) if SETTINGS.theme_dark else (225,225,230)
    PANEL = (40,40,50) if SETTINGS.theme_dark else (245,245,250)
    FG = (230,230,240) if SETTINGS.theme_dark else (20,20,25)
    ACCENT = (90,120,255)

    DRAW = (255,255,255)
    ERASE = (0,0,0)

    draw_size = 420
    draw_x = WIDTH//2 - draw_size//2
    draw_y = 160
    drawing_surface = pygame.Surface((draw_size, draw_size))
    drawing_surface.fill(ERASE)

    font = pygame.font.SysFont('Arial', 34)
    sfont = pygame.font.SysFont('Arial', 20)

    # UI Buttons
    buttons = []
    buttons.append(Button((40, HEIGHT-100, 140, 44), "Clear (C)", sfont, ACCENT, (120,150,255), (255,255,255)))
    buttons.append(Button((200, HEIGHT-100, 160, 44), "Undo (U)", sfont, ACCENT, (120,150,255), (255,255,255)))
    buttons.append(Button((380, HEIGHT-100, 180, 44), "Save PNG (S)", sfont, ACCENT, (120,150,255), (255,255,255)))
    buttons.append(Button((580, HEIGHT-100, 200, 44), "Explain (E)", sfont, ACCENT, (120,150,255), (255,255,255)))
    buttons.append(Button((800, HEIGHT-100, 250, 44), "Quit to Menu (Q)", sfont, ACCENT, (120,150,255), (255,255,255)))

    eraser_toggle = Toggle((40, 120, 60, 28), "Eraser", sfont, initial=False)

    drawing = False
    last_pos = None
    brush = 22
    stack = deque(maxlen=20)
    prediction = "Draw a digit"
    confidence = 0.0
    dist = None
    heatmap = None
    last_infer_time = 0

    clock = pygame.time.Clock()

    def push_state():
        stack.append(drawing_surface.copy())

    while True:
        mouse = pygame.mouse.get_pos()
        for b in buttons: b.update_hover(mouse)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            if eraser_toggle.toggle_on_click(event, mouse):
                pass
            if event.type == KEYDOWN:
                if event.key == K_c:
                    drawing_surface.fill(ERASE); prediction="Draw a digit"; confidence=0.0; dist=None; heatmap=None
                elif event.key == K_q:
                    pygame.quit(); return
                elif event.key == K_UP:
                    brush = min(48, brush+2)
                elif event.key == K_DOWN:
                    brush = max(4, brush-2)
                elif event.key == K_u and len(stack)>0:
                    drawing_surface.blit(stack.pop(), (0,0))
                elif event.key == K_s:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    pygame.image.save(drawing_surface, f"pad_{ts}.png")
                elif event.key == K_e and dist is not None and confidence>0.7 and heatmap is not None:
                    # simple explain popup
                    popup = pygame.Surface((640, 360))
                    popup.fill(PANEL)
                    draw_text(popup, "Prediction Explanation", (20, 20), font, FG)
                    draw_text(popup, f"Prediction: {prediction}  ({confidence*100:.1f}%)", (20, 70), sfont, FG)
                    # heatmap
                    hm = (np.uint8(255*cv2.resize(heatmap,(120,120))))
                    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
                    hm_surf = pygame.surfarray.make_surface(np.transpose(hm,(1,0,2)))
                    popup.blit(hm_surf, (40, 110))
                    draw_text(popup, "Attention Heatmap", (40, 240), sfont, FG)
                    # processed
                    # regenerate processed preview from current canvas
                    arr = pygame.surfarray.array3d(drawing_surface)
                    arr = np.transpose(arr, (1,0,2))
                    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                    resz = cv2.resize(gray, (28,28))
                    _,_,_, proc = enhanced_predict(model, resz, return_all=True)
                    proc_show = np.uint8(255*cv2.resize(proc,(120,120)))
                    proc_show = cv2.cvtColor(proc_show, cv2.COLOR_GRAY2RGB)
                    proc_surf = pygame.surfarray.make_surface(np.transpose(proc_show,(1,0,2)))
                    popup.blit(proc_surf, (220,110))
                    draw_text(popup, "Processed 28x28", (220, 240), sfont, FG)
                    # instruction
                    draw_text(popup, "Press any key to close", (20, 310), sfont, FG)
                    screen.blit(popup, (WIDTH//2-320, HEIGHT//2-180))
                    pygame.display.flip()
                    waiting=True
                    while waiting:
                        for e2 in pygame.event.get():
                            if e2.type in (KEYDOWN, MOUSEBUTTONDOWN, QUIT):
                                waiting=False
                                if e2.type==QUIT: pygame.quit(); return

            if event.type == MOUSEBUTTONDOWN and event.button==1:
                if any(b.clicked(event) for b in buttons):
                    # map button actions
                    if buttons[0].hover: # clear
                        drawing_surface.fill(ERASE); prediction="Draw a digit"; confidence=0.0; dist=None; heatmap=None
                    elif buttons[1].hover: # undo
                        if len(stack)>0: drawing_surface.blit(stack.pop(), (0,0))
                    elif buttons[2].hover: # save
                        ts = time.strftime("%Y%m%d_%H%M%S"); pygame.image.save(drawing_surface, f"pad_{ts}.png")
                    elif buttons[3].hover: # explain
                        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_e))
                    elif buttons[4].hover: # quit
                        pygame.quit(); return
                else:
                    # start drawing
                    if (draw_x <= mouse[0] <= draw_x+draw_size) and (draw_y <= mouse[1] <= draw_y+draw_size):
                        drawing = True
                        push_state()
                        last_pos = (mouse[0]-draw_x, mouse[1]-draw_y)
            if event.type == MOUSEBUTTONUP and event.button==1:
                drawing=False; last_pos=None
            if event.type == MOUSEMOTION and drawing:
                if (draw_x <= mouse[0] <= draw_x+draw_size) and (draw_y <= mouse[1] <= draw_y+draw_size):
                    cur = (mouse[0]-draw_x, mouse[1]-draw_y)
                    color = ERASE if eraser_toggle.value else DRAW
                    if last_pos:
                        pygame.draw.line(drawing_surface, color, last_pos, cur, brush)
                    last_pos = cur

        # background
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (20,20, WIDTH-40, 110), border_radius=12)
        draw_text(screen, "Drawing Pad", (32, 28), font, FG)
        draw_text(screen, "UP/DOWN: Brush | Toggle Eraser | C: Clear | U: Undo | S: Save | E: Explain", (32, 72), sfont, FG)
        eraser_toggle.draw(screen)
        draw_text(screen, f"Brush: {brush}", (130, 120), sfont, FG)

        # drawing area
        pygame.draw.rect(screen, FG, (draw_x-3, draw_y-3, draw_size+6, draw_size+6), 3, border_radius=8)
        screen.blit(drawing_surface, (draw_x, draw_y))

        # periodic inference
        now = time.time()
        if now - last_infer_time > 0.25:
            arr = pygame.surfarray.array3d(drawing_surface)
            arr = np.transpose(arr, (1,0,2))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            resz = cv2.resize(gray, (28,28))
            if np.any(resz < 255):
                prediction, confidence, dist, proc = enhanced_predict(model, resz, return_all=True)
                heatmap = generate_grad_cam(model, proc) if confidence>0.7 else None
            else:
                prediction="Draw a digit"; confidence=0.0; dist=None; heatmap=None
            last_infer_time = now

        # side panel: prediction
        pygame.draw.rect(screen, PANEL, (20, HEIGHT-220, WIDTH-40, 110), border_radius=12)
        draw_text(screen, f"Prediction: {prediction}", (32, HEIGHT-210), font, FG)
        # confidence bar
        pygame.draw.rect(screen, (80,80,90), (32, HEIGHT-160, 240, 20), border_radius=6)
        pygame.draw.rect(screen, (90,200,120), (32, HEIGHT-160, int(240*confidence), 20), border_radius=6)
        pygame.draw.rect(screen, (30,30,30), (32, HEIGHT-160, 240, 20), 2, border_radius=6)
        draw_text(screen, f"{confidence*100:.1f}%", (280, HEIGHT-163), sfont, FG)

        # probability list
        if dist is not None:
            draw_text(screen, "Probabilities:", (360, HEIGHT-190), sfont, FG)
            for i, p in enumerate(dist[:10]):
                col = (120,220,160) if prediction != "Draw a digit" and i==int(prediction) else (170,170,180)
                pygame.draw.rect(screen, col, (360, HEIGHT-160 + i*18, int(160*p), 14), border_radius=4)
                draw_text(screen, f"{i}: {p*100:.1f}%", (530, HEIGHT-163 + i*18), sfont, FG)

        # heatmap preview
        if heatmap is not None and confidence>0.7:
            hm = np.uint8(255*cv2.resize(heatmap,(120,120)))
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            hm_surf = pygame.surfarray.make_surface(np.transpose(hm,(1,0,2)))
            screen.blit(hm_surf, (WIDTH-160, 30))
            draw_text(screen, "Attention", (WIDTH-155, 158), sfont, FG)

        for b in buttons: b.draw(screen)

        pygame.display.flip()
        clock.tick(60)

# -------------------- Mode Selection + Settings --------------------
def settings_panel():
    init_pygame()
    W,H = 700, 520
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("Settings")

    font = pygame.font.SysFont('Arial', 32)
    sfont = pygame.font.SysFont('Arial', 22)

    dec = Button((60, 140, 40, 40), "-", sfont, (90,120,255), (120,150,255), (255,255,255))
    inc = Button((250, 140, 40, 40), "+", sfont, (90,120,255), (120,150,255), (255,255,255))

    res_down = Button((60, 220, 60, 40), "-", sfont, (90,120,255), (120,150,255), (255,255,255))
    res_up   = Button((220, 220, 60, 40), "+", sfont, (90,120,255), (120,150,255), (255,255,255))

    roi_down = Button((60, 300, 60, 40), "-", sfont, (90,120,255), (120,150,255), (255,255,255))
    roi_up   = Button((220, 300, 60, 40), "+", sfont, (90,120,255), (120,150,255), (255,255,255))

    theme_toggle = Toggle((60, 380, 60, 28), "Dark Theme", sfont, SETTINGS.theme_dark)
    back_btn = Button((W-160, H-70, 120, 44), "Back", sfont, (90,120,255), (120,150,255), (255,255,255))

    while True:
        mouse = pygame.mouse.get_pos()
        for b in (dec,inc,res_down,res_up,roi_down,roi_up,back_btn):
            b.update_hover(mouse)

        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit(); return
            if dec.clicked(e):
                SETTINGS.camera_index = max(0, SETTINGS.camera_index-1)
            if inc.clicked(e):
                SETTINGS.camera_index += 1
            if res_down.clicked(e):
                SETTINGS.cam_width = max(640, SETTINGS.cam_width-160)
                SETTINGS.cam_height = max(360, SETTINGS.cam_height-90)
            if res_up.clicked(e):
                SETTINGS.cam_width = min(1920, SETTINGS.cam_width+160)
                SETTINGS.cam_height = min(1080, SETTINGS.cam_height+90)
            if roi_down.clicked(e):
                SETTINGS.roi_box = max(140, SETTINGS.roi_box-20)
            if roi_up.clicked(e):
                SETTINGS.roi_box = min(400, SETTINGS.roi_box+20)
            if theme_toggle.toggle_on_click(e, mouse):
                SETTINGS.theme_dark = theme_toggle.value
            if back_btn.clicked(e):
                pygame.quit(); return

        screen.fill((25,25,30))
        draw_text(screen, "Settings", (30, 30), font, (255,255,255))
        draw_text(screen, f"Camera Index: {SETTINGS.camera_index}", (60, 110), sfont, (255,255,255))
        dec.draw(screen); inc.draw(screen)

        draw_text(screen, f"Resolution: {SETTINGS.cam_width}x{SETTINGS.cam_height}", (60, 190), sfont, (255,255,255))
        res_down.draw(screen); res_up.draw(screen)

        draw_text(screen, f"ROI Size: {SETTINGS.roi_box}", (60, 270), sfont, (255,255,255))
        roi_down.draw(screen); roi_up.draw(screen)

        theme_toggle.draw(screen)
        back_btn.draw(screen)
        pygame.display.flip()

def select_mode(model, x_test, y_test):
    init_pygame()
    WIDTH, HEIGHT = 860, 620
    WHITE = (255,255,255); BLACK=(0,0,0)
    BLUE=(90,120,255); LIGHT=(120,150,255); BG=(25,25,30)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Digit Recognizer - Mode Selection')
    title_font = pygame.font.SysFont('Arial', 48)
    button_font = pygame.font.SysFont('Arial', 28)
    info_font = pygame.font.SysFont('Arial', 20)

    class ModeButton(Button):
        def __init__(self, x, y, w, h, text, mode):
            super().__init__((x,y,w,h), text, button_font, BLUE, LIGHT, WHITE, 12)
            self.mode = mode

    btns = [
        ModeButton(WIDTH//2-160, 200, 320, 56, "Webcam Mode", "webcam"),
        ModeButton(WIDTH//2-160, 280, 320, 56, "Drawing Pad Mode", "drawing"),
        ModeButton(WIDTH//2-160, 360, 320, 56, "Evaluate Model", "evaluate"),
        ModeButton(WIDTH//2-160, 440, 320, 56, "Settings", "settings"),
        ModeButton(WIDTH//2-160, 520, 320, 56, "Quit", "quit")
    ]

    running = True
    while running:
        mouse = pygame.mouse.get_pos()
        for b in btns: b.update_hover(mouse)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN and event.button==1:
                for b in btns:
                    if b.hover:
                        if b.mode == "webcam":
                            pygame.quit(); webcam_mode(model); init_pygame(); screen = pygame.display.set_mode((WIDTH, HEIGHT))
                        elif b.mode == "drawing":
                            pygame.quit(); drawing_pad_mode(model); init_pygame(); screen = pygame.display.set_mode((WIDTH, HEIGHT))
                        elif b.mode == "evaluate":
                            evaluate_model(model, x_test, y_test)
                            screen.fill(BG)
                            done = title_font.render("Evaluation Complete!", True, WHITE)
                            screen.blit(done, (WIDTH//2-done.get_width()//2, HEIGHT//2-40))
                            info = info_font.render("Check confusion_matrix.png, misclassified_examples.png, classification_report.txt", True, WHITE)
                            screen.blit(info, (WIDTH//2-info.get_width()//2, HEIGHT//2+20))
                            pygame.display.flip(); pygame.time.wait(1800)
                        elif b.mode == "settings":
                            pygame.quit(); settings_panel(); init_pygame(); screen = pygame.display.set_mode((WIDTH, HEIGHT))
                        elif b.mode == "quit":
                            running = False

        screen.fill(BG)
        title_surf = title_font.render("Digit Recognizer", True, WHITE)
        screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 60))
        for b in btns: b.draw(screen)
        info = info_font.render("Choose a mode. Tip: tune Settings first if your webcam index is different.", True, WHITE)
        screen.blit(info, (WIDTH//2 - info.get_width()//2, 150))
        pygame.display.flip()

    pygame.quit()

# -------------------- Main --------------------
def main():
    model_path = 'advanced_digit_model.keras'
    print("Getting MNIST data...")
    x_train, y_train, x_test, y_test = get_mnist_data()

    x_test_eval = x_test.astype('float32')/255.0
    x_test_eval = np.expand_dims(x_test_eval, -1)

    try:
        model = tf.keras.models.load_model(model_path)
        print('Loaded saved advanced model.')
    except Exception as e:
        print("Training advanced model (this can take a while)...")
        model, _ = train_model(x_train, y_train, x_test, y_test)
        print("Saving model...")
        model.save(model_path)
        print("Advanced model trained and saved!")

    # warmup (faster first inference)
    _ = model.predict(np.zeros((1,28,28,1), dtype=np.float32), verbose=0)

    print("Starting digit recognizer...")
    select_mode(model, x_test_eval, y_test)

if __name__ == '__main__':
    main()
