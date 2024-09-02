from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

#加载模型
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
#要预测的图片路径
IMAGE_PATH = "C:\\Users\\hjk13\\Desktop\\Cover detection\\images\\cover5.jpg"
#要预测的类别提示，可以输入多个类中间用英文句号隔开
TEXT_PROMPT = "Good manhole cover . Bad manhole cover"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
#保存预测的图片,保存到outputs文件夹中，名称为annotated_image.jpg
cv2.imwrite("outputs/annotated_image5.2.jpg", annotated_frame)
