Here are some draft results paragraphs describing neural analyses on the dot-prediction task, spelling out some results that I think could be both neurally and cognitively interesting.

Quibbles

*   The two parts are currently interdependent. It’s far too soon to know for certain but it might be good to start pondering whether these are two separate stories/projects/papers or one story/project/paper.
    

**Distilled claims: dot-prediction**
====================================

1.  Humans use compositional representations built from reusable primitives to reason about abstract visual patterns._Alternatives:_ continuous embedding space; exemplars
    
    1.  _Behavioral detail:_ Click behavior model fit by LoT vs. other models. Saccade behavior fit by LoT vs. other models.
        
    2.  _Neural detail:_ vlPFC exhibits a distributed representation of _potential_ visual patterns that is compositional (independent components in the population’s neural activation correspond to primitives in a compositional model).
        
2.  This abstract reasoning is implemented by discrete steps of hypothesis revision over a structured hypothesis space._Alternatives:_ continuous evidence accumulation; drift-diffusion model
    
    1.  _Behavioral detail:_ Highly nonlinear trial trajectories, with particular points triggering massive belief updates (model-free: RT collapse and accuracy spike; model-based: entropy reduction).
        
    2.  _Neural detail:_ A network linking m/oPFC and vlPFC exhibit a characteristic behavior at these punctuated moments of belief transition.
        
3.  Subjects rationally explore the hypothesis space through both behavior and active sensing (eye movements).
    
    1.  _Behavioral detail:_ Click behaviors are “rational” under an LoT model. Saccades reflect both 1) backward-looking evaluation of top-k hypotheses and 2) forward-looking evaluation of their predictions.
        
    2.  _Neural detail:_ Medial and orbital frontal regions are implicated in both components of this hypothesis exploration. First, they coordinate both click behaviors and saccades driven by beliefs about the latent pattern. Second, they process information resulting from behaviors and saccades, and show traces of model-based belief updating.
        
    

Other ideas

*   Sample-based representation of possibilities, evoked maybe in the first or second click of a trial
    
*   Entropy reduction variants
    
    *   going from uncertainty over ~2 to certainty on 1
        
    *   going from “no idea” to certainty on 1
        
*   Model fit of LoT vs behavior between subjects, and concordant neural difference? (But we need more subjects)
    
*   Click-by-click hypothesis elimination in the workspace