from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from PIL import Image, ImageDraw
import requests
import torch
import os

# 检查 MPS 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")
# Initialize KOSMOS-2 model and processor globally
# processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("/Volumes/blockdata/ai/models/kosmos2/")
model = AutoModelForVision2Seq.from_pretrained("/Volumes/blockdata/ai/models/kosmos2/", device_map={"": device})

# model = AutoModelForCausalLM.from_pretrained(
#     "/Volumes/blockdata/ai/models/Florence-2-large",
#     # torch_dtype=torch_dtype,
#     trust_remote_code=True,
#     device_map={"": device}
# )
# processor = AutoProcessor.from_pretrained(
#     "/Volumes/blockdata/ai/models/Florence-2-large",
#     trust_remote_code=True
# )
def process_image(image_url,prompt):
    # Load image from URL

    image = Image.open(requests.get(image_url, stream=True).raw)
    # prompt = "<OD>"
    # Prepare inputs for the model
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Autoregressively generate completion
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Extract entities
    processed_text, entities = processor.post_process_generation(generated_text)
    
    # generated_ids = model.generate(
    #     input_ids=inputs["input_ids"],
    #     pixel_values=inputs["pixel_values"],
    #     max_new_tokens=1024,
    #     num_beams=3,
    #     do_sample=False,
    # )
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
    
    return image, processed_text, entities


def index(request):
    # Load image from URL
    image_url = "https://hf-mirror.com/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
    default_url = "https://tse3-mm.cn.bing.net/th/id/OIP-C.UfH-QuSzFmt5UBT6soE10gHaFj?rs=1&pid=ImgDetMain"
    default_url = "https://img95.699pic.com/photo/50126/6811.jpg_wh860.jpg"
    default_prompt = "<grounding>An image of"
    default_prompt = "<grounding>Question: How many people are here? Answer:"

    image_url = request.GET.get("image_url", default_url)
    prompt = request.GET.get("prompt", default_prompt)
    image, processed_text, entities = process_image(image_url,prompt)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for entity, _, box in entities:
        box = [round(i, 2) for i in box[0]]
        x1, y1, x2, y2 = tuple(box)
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
        draw.text(xy=(x1, y1), text=entity)

    # Save the annotated image
    image_dir = os.path.join(
        settings.BASE_DIR, "kosmos", "static", "kosmos_app", "images"
    )
    os.makedirs(image_dir, exist_ok=True)

    annotated_image_path = 'static/kosmos_app/images/annotated_image.png'
    image.save(os.path.join(settings.BASE_DIR, "kosmos", annotated_image_path))

    # Render the result in HTML
    return render(
        request,
        "kosmos_app/index.html",
        {
            "annotated_image_path": annotated_image_path,
            "annotated_image_text": processed_text + "\n" + str(entities),
        },
    )


@csrf_exempt
def kosmos_api(request):
    if request.method == "POST":

        # Get image from request
        default_url = "https://tse3-mm.cn.bing.net/th/id/OIP-C.UfH-QuSzFmt5UBT6soE10gHaFj?rs=1&pid=ImgDetMain"
        default_prompt = "<grounding>An image of"

        image_url = request.POST.get("image_url", default_url)
        prompt = request.POST.get("prompt", default_prompt)

        # image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        # Extract entities
        image, processed_text, entities = process_image(image_url, prompt)

        # Prepare response
        response_data = {"processed_text": processed_text, "entities": entities}

        return JsonResponse(response_data)
    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)
