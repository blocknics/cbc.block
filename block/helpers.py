from dolfin import MPI

class MpiNotSupported(Exception):
    pass

def supports_mpi(yesno, msg=None, size=MPI.size(MPI.comm_world)):
    if not yesno and size > 1:
        import os, sys
        if os.environ.get('BLOCK_REGRESSION_MPIRUN'):
            if MPI.comm_world.rank == 0:
                print(f'skip {sys.argv[0]}: {msg or "MPI not supported"}', file=sys.stderr)
            sys.exit(0)
        else:
            raise MpiNotSupported(msg)
