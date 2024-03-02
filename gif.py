from ddpm import trainer, Tester

trainer.sample(output_name='output6', sample_count=4)

trainer.sample_gif(
    output_name="output8",
    sample_count=1,
    save_path=r"/home/gamma/home/gamma/Workbenches/D/saved_outputs"
)

tester = Tester(device='cuda')
tester.test_unet()
tester.test_attention()
tester.test_jit()
