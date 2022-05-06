import os, gc
from custom_parser import *
from VQGAN_and_CLIP import *

# This will need to be moved to the input file
NN_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'vqgan_f16_16384')
CONFIG_FILE = 'config.yaml'
CHECKPOINT_FILE = 'model.ckpt'
INTERVAL_IMAGE = 25

# Input file parsing
INPUT_PATH = 'C:\\WORKSPACES\\ZINKY\\GenerativeNetworks\\InputFiles\\scary_colorful_images_feed.yaml'
settings_dictionary = parse_input_yaml(INPUT_PATH)
list_of_sims = generate_simulations(settings_dictionary, max_number = 10)

# Run the generator

for sim_index, sim in enumerate(list_of_sims):
    # Initialisation
    print(f' ImageGeneration.run :: running image gen {sim_index + 1} of {len(list_of_sims)}')
    sim.generate_path()
    sim.save_sim_settings()
    # creating the input structure for the arguments
    args = argparse.Namespace(
        prompt = sim.full_string,
        objective_image = sim.objective_image,
        noise_prompt_seeds = [],
        noise_prompt_weights = [],
        size = [sim.width, sim.height],
        init_image = sim.initial_image,
        init_weight = sim.initial_image_weight,
        clip_model = 'ViT-B/32',
        vqgan_model = 'vqgan_imagenet_f16_16384',
        vqgan_config = os.path.join(NN_CHECKPOINT_PATH, CONFIG_FILE),
        vqgan_checkpoint = os.path.join(NN_CHECKPOINT_PATH, CHECKPOINT_FILE),
        step_size = sim.learning_rate,
        cutn = 64,
        cut_pow = 1.,
        display_freq = INTERVAL_IMAGE,
        seed = sim.seed,
        max_iterations = sim.max_iterations,
    )
    descriptive_model_name = list_of_models[args.vqgan_model]
    # Switches based on the model selected
    if args.vqgan_model == "gumbel_8192":
        is_gumbel = True
    else:
        is_gumbel = False
    if args.seed == -1:
        seed = None

    if args.init_image != [] or args.objective_image != []:
        input_images = True
    else:
        input_images = False

    # Device selection and clean memory
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Quick summary of what we are running now. 
    if args.prompt :
        print('Using texts :: ', args.prompt)
    if args.init_image != []:
        print('Using initial images :: ', ' '.join(args.init_image))
    if args.objective_image != []:
        print('Using objective images :: ', ' '.join(args.objective_image))
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

    if args.init_image != []:
        pil_image = Image.open(args.init_image[0]).convert('RGB')
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

    for keyword in [args.prompt]:
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
        if args.prompt:
            imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', args.prompt, {"prop_array_is_ordered": True, "prop_value_is_array": True})
        else:
            imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', 'None', {"prop_array_is_ordered": True, "prop_value_is_array": True})
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'i', str(i), {"prop_array_is_ordered": True, "prop_value_is_array": True})
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'model', descriptive_model_name, {"prop_array_is_ordered": True, "prop_value_is_array": True})
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'seed', str(seed), {"prop_array_is_ordered": True, "prop_value_is_array": True})
        imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'input_images', str(input_images), {"prop_array_is_ordered": True, "prop_value_is_array": True})
        # for frases in args.prompts:
        #    imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'Prompt' ,frases, {"prop_array_is_ordered":True, "prop_value_is_array":True})
        imagen.close()

    def add_stegano_data(filename):
        data = {
            "title": args.prompt,
            "notebook": "VQGAN+CLIP",
            "i": i,
            "model": descriptive_model_name,
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
            filename = os.path.join(sim.save_path, f'{i:05}.png')
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
            if i == args.max_iterations:
                break
            i += 1
    except KeyboardInterrupt:
        pass

# Garbage collection
gc.collect()