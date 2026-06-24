// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import * as React from 'react';

import { cn } from '@/lib/utils';

/** Surface panel — themed by the bespoke `.card` design-system class. */
const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('card', className)} {...props} />
  ),
);
Card.displayName = 'Card';

export { Card };
