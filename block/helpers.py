class MpiNotSupported(Exception):
    pass

def supports_mpi(yesno, msg=None):
    if not yesno:
        from dolfin import MPI
        if MPI.size(MPI.comm_world) > 1:
            import os, sys
            if os.environ.get('BLOCK_REGRESSION_MPIRUN'):
                if MPI.comm_world.rank == 0:
                    print(f'skip {sys.argv[0]}: {msg or "MPI not supported"}', file=sys.stderr)
                sys.exit(0)
            else:
                raise MpiNotSupported(msg)
