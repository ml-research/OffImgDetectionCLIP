from main.experiments import torch, parser, RTPT, Image
from main.experiments import eval_model_
from main.experiments import ClipSingleSimModel

torch.set_num_threads(6)
args = parser.parse_args()

# Create RTPT object and start the RTPT tracking
rtpt = RTPT(name_initials='Kersting', experiment_name='CrazyStuff', max_iterations=1)
rtpt.start()

model = ClipSingleSimModel(args, label='offending') # toxic, negative, unpleasant


sample_names = "b10_p133_8 " \
"b10_p138_14 " \
"b11_p158_17 " \
"b11_p162_15 " \
"b11_p167_11 " \
"b11_p167_16 " \
"b11_p167_18 " \
"b11_p171_14 " \
"b11_p172_15 " \
"b11_p176_7 " \
"b14_p253_4 " \
"b14_p254_12 " \
"b5_p80_6"
sample_names = sample_names.split(' ')

for sample_name in sample_names:
    file_name = 'b15_p369_9'
    file_path = f"/workspace/datasets/SMID_images_400px/img/{file_name}.jpg"

    x = model.preprocess(Image.open(file_path)).unsqueeze(0)


    x1 = ''
    eval_model_(x=x, model=model, file_name=file_name)

