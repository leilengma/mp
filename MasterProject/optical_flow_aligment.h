#ifndef OPTICAL_FLOW_ALIGMENT_H_INCLUDED
#define OPTICAL_FLOW_ALIGMENT_H_INCLUDED

#include "frame.h"

namespace rrlib{
    namespace oft{
        void motion_alignment(FramePtr ref_frame,FramePtr cur_frame);
    }
}


#endif // OPTICAL_FLOW_ALIGMENT_H_INCLUDED
