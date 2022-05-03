import os, gc
# from custom_parser import *
from VQGAN_and_CLIP import *

results_path = os.path.join(os.path.dirname(__file__), os.pardir, 'Results')
folder_counter_start = 19
resources_path = os.path.join(os.path.dirname(__file__), os.pardir, 'vqgan_f16_16384')
config_file = 'config.yaml'
checkpoint_file = 'model.ckpt'

# Art Generator Parameters
# Examples text_prompts = ['unreal engine', 'center of the universe']
descriptions = [
    # "zombie panda illustration in saturated colors",
    # "laughing panda illustation in saturated colors",
    # "screaming kangaroo illustration in saturated colors",
    # "scary koala illustration in saturated colors",
    # "ferocious gorilla illustration in saturated colors",
    # "thunder tiger illustration in saturated colors",
    # "terrifying lemurs illustration in saturated colors",
    # "terrifying octopus illustration in saturated colors",
    # "terrifying peacock illustration in saturated colors",
    # "screaming hippopotamus illustration in saturated colors",
    'oil on canvas'
]
# Acceptable (500x500), Twitter (600x335), 
height = 500  # @param {type:"number"}
width = 500  # @param {type:"number"}
# @param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_1024", "wikiart_16384", "coco", "faceshq", "sflckr", "ade20k", "ffhq", "celebahq", "gumbel_8192"]
model = "vqgan_imagenet_f16_16384"
interval_image = 50  # @param {type:"number"}
initial_image = [
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test06\large-Panda-photo.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test07\giantpandabamboodiet.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test08\kangaroo-stock.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test09\Koala.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test10\Gorilla1.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test11\Tiger1.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test12\Ring-tailed-Lemurs-540x350-1.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test13\Octopus1.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test14\Peacock1.jpg'),
    # os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test15\Hippo1.jpg'),
    os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test18\Anna.jpeg')
]
objective_image = [os.path.abspath('C:\WORKSPACES\ZINKY\GenerativeNetworks\Results\Test18\Anna.jpeg')]  # @param {type:"string"}
seed = 524  # @param {type:"number"}
max_iterations = 2000  # @param {type:"number"}
init_image_weight = 0

nombre_model = nombres_modelos[model]

args = argparse.Namespace(
    prompts = descriptions,
    objective_image = objective_image,
    noise_prompt_seeds = [],
    noise_prompt_weights = [],
    size = [width, width],
    init_image = initial_image,
    init_weight = init_image_weight,
    clip_model = 'ViT-B/32',
    vqgan_config = os.path.join(resources_path, config_file),
    vqgan_checkpoint = os.path.join(resources_path, checkpoint_file),
    step_size = 0.02,
    cutn = 64,
    cut_pow = 1.,
    display_freq = interval_image,
    seed = seed,
)


if model == "gumbel_8192":
    is_gumbel = True
else:
    is_gumbel = False

if seed == -1:
    seed = None

if initial_image == "None":
    initial_image = None
# elif initial_image and initial_image.lower().startswith("http"):
#     initial_image = download_img(initial_image)


if objective_image == "None" or not objective_image:
    objective_image = []
# else:
#     objective_image = objective_image.split("|")
#     objective_image = [image.strip() for image in objective_image]

if initial_image != [] or objective_image != []:
    input_images = True

# Run the generator

for test_index, prompt in enumerate(args.prompts):
    # Device selection and clean memory
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    if prompt:
        print('Using texts:', prompt)
    if objective_image:
        print('Using image prompts:', objective_image)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    if is_gumbel:
        e_dim = model.quantize.embedding_dim
    else:
        e_dim = model.quantize.e_dim

    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    if is_gumbel:
        n_toks = model.quantize.n_embed
    else:
        n_toks = model.quantize.n_e

    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    if is_gumbel:
        z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
    else:
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if args.init_image:
        pil_image = Image.open(args.init_image[test_index]).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, * \
            _ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(
            n_toks, [toksY * toksX], device=device), n_toks).float()
        if is_gumbel:
            z = one_hot @ model.quantize.embed.weight
        else:
            z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    for keyword in [prompt]:
        txt, weight, stop = parse_prompt(keyword)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for obj_image in args.objective_image:
        path, weight, stop = parse_prompt(obj_image)
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    def synth(z):
        if is_gumbel:
            z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
        else:
            z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)

        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

    def add_xmp_data(nombrefichero):
        imagen = ImgTag(filename=nombrefichero)
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'creator', 'VQGAN+CLIP', {"prop_array_is_ordered": True, "prop_value_is_array": True})
        if args.prompts:
            imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', " | ".join(args.prompts), {"prop_array_is_ordered": True, "prop_value_is_array": True})
        else:
            imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', 'None', {"prop_array_is_ordered": True, "prop_value_is_array": True})
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'i', str(i), {"prop_array_is_ordered": True, "prop_value_is_array": True})
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'model', nombre_model, {"prop_array_is_ordered": True, "prop_value_is_array": True})
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'seed', str(seed), {"prop_array_is_ordered": True, "prop_value_is_array": True})
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'input_images', str(input_images), {"prop_array_is_ordered": True, "prop_value_is_array": True})
        # for frases in args.prompts:
        #    imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'Prompt' ,frases, {"prop_array_is_ordered":True, "prop_value_is_array":True})
        imagen.close()

    def add_stegano_data(filename):
        data = {
            "title": " | ".join(args.prompts) if args.prompts else None,
            "notebook": "VQGAN+CLIP",
            "i": i,
            "model": nombre_model,
            "seed": str(seed),
            "input_images": input_images
        }
        lsb.hide(filename, json.dumps(data)).save(filename)

    @torch.no_grad()
    def checkin(i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = synth(z)
        TF.to_pil_image(out[0].cpu()).save('progress.png')
        add_stegano_data('progress.png')
        # add_xmp_data('progress.png')
        # display.display(display.Image('progress.png'))

    def ascend_txt():
        global i
        out = synth(z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
        result = []
        if args.init_weight:
            result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
        for prompt in pMs:
            result.append(prompt(iii))
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        if i % 20 == 0:
            filename = os.path.join(results_path, ''.join(['Test', f'{folder_counter_start:02}']), f'{i:05}.png')
            imageio.imwrite(filename, np.array(img))
            add_stegano_data(filename)
            # add_xmp_data(filename)
        return result

    def train(i):
        opt.zero_grad()
        lossAll = ascend_txt()
        if i % args.display_freq == 0:
            checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

    i = 0
    try:
        while True:
            train(i)
            if i == max_iterations:
                break
            i += 1
    except KeyboardInterrupt:
        pass
    
    folder_counter_start += 1
# Garbage collection
gc.collect()