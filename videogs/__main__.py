import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='RGB video -> COLMAP -> hierarchical semantic incremental 3DGS')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','prepare'):
        p=sub.add_parser(name);p.set_defaults(prepare_only=name=='prepare')
        p.add_argument('--video',required=True);p.add_argument('--output',required=True)
        p.add_argument('--fps',type=float,default=3.);p.add_argument('--max-frames',type=int,default=180)
        p.add_argument('--extract-width',type=int,default=960);p.add_argument('--width',type=int,default=640)
        p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float)
        p.add_argument('--rotate',type=float);p.add_argument('--blur-threshold',type=float,default=0.)
        p.add_argument('--duplicate-threshold',type=float,default=1.)
        p.add_argument('--camera-model',choices=['SIMPLE_RADIAL','OPENCV','OPENCV_FISHEYE','PINHOLE'],default='SIMPLE_RADIAL')
        p.add_argument('--camera-params',default='');p.add_argument('--overlap',type=int,default=12)
        p.add_argument('--min-registration',type=float,default=.75);p.add_argument('--threads',type=int,default=4)
        p.add_argument('--stages',type=int,default=6);p.add_argument('--steps',type=int,default=2400)
        p.add_argument('--cap',type=int,default=500000);p.add_argument('--factors',choices=['000','001','010','011','100','101','110','111'],default='111')
        p.add_argument('--important',default='monitor,keyboard,mouse,cup,bottle,book,laptop')
        p.add_argument('--semantic-densify',type=int,choices=[0,1],default=1)
        p.add_argument('--background-prune-fraction',type=float,default=.03)
        p.add_argument('--boundary-boost',type=float,default=.75);p.add_argument('--object-boost',type=float,default=.5)
        p.add_argument('--feature-weight',type=float,default=.01);p.add_argument('--semantic-every',type=int,default=10)
        p.add_argument('--seed',type=int,default=7);p.add_argument('--teacher-stride',type=int,default=3)
        p.add_argument('--checkpoint',default=str(Path(__file__).resolve().parents[1]/'checkpoints/sam2.1_hiera_tiny.pt'))
        from hglab.refinement import add_stability_arguments
        add_stability_arguments(p)
    p=sub.add_parser('demo');p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=24.)
    sub.add_parser('doctor')
    p=sub.add_parser('serve');p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=8770)
    args=parser.parse_args()
    if args.command in ('run','prepare'):
        from .pipeline import run
        run(args)
    elif args.command=='doctor':
        from .pipeline import doctor
        doctor()
    elif args.command=='demo':
        from .demo import make_demo
        make_demo(args.output,args.seconds)
    else:
        import http.server,functools
        from .server import PlaybackHandler
        handler=functools.partial(PlaybackHandler,directory=str(Path(args.output).resolve()))
        print(f'Open http://127.0.0.1:{args.port}/viewer.html',flush=True)
        http.server.ThreadingHTTPServer(('127.0.0.1',args.port),handler).serve_forever()

if __name__=='__main__':main()
